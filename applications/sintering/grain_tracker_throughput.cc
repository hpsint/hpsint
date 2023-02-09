// Test performance of the grain tracker

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#ifdef LIKWID_PERFMON
#undef LIKWID_PERFMON
#endif

#include <pf-applications/base/performance.h>
#include <pf-applications/base/revision.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/sintering/initial_values_hypercube.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>

using namespace dealii;
using namespace Sintering;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  AssertThrow(
    argc >= 2,
    ExcMessage(
      "Maximum number of particles per row of the hypercube has to be specified"));

  const unsigned int max_grains_per_row = atoi(argv[1]);

  AssertThrow(max_grains_per_row > 0,
              ExcMessage(
                "Maximum number of particles per row has to be grater than 1"));

  constexpr bool test_detect_grains   = true;
  constexpr bool test_initial_setup   = true;
  constexpr bool test_track           = true;
  constexpr bool test_remap_dependent = true;
  constexpr bool test_remap_cycle     = true;

  using Number = double;
  using BlockVectorType =
    LinearAlgebra::distributed::DynamicBlockVector<Number>;

  // Approximation
  constexpr unsigned int dim = SINTERING_DIM;

  MappingQ1<dim> mapping;

  const unsigned int        fe_degree = 1;
  FE_Q<dim>                 fe(fe_degree);
  AffineConstraints<Number> constraints;

  ConvergenceTable table;

  for (unsigned int n_grains_per_row = 2;
       n_grains_per_row <= max_grains_per_row;
       ++n_grains_per_row)
    {
      const unsigned int n_grains = std::pow(n_grains_per_row, dim);

      std::array<unsigned int, dim> n_grains_dir;
      n_grains_dir.fill(n_grains_per_row);

      parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
      DoFHandler<dim>                           dof_handler(tria);

      // Geometry
      const double radius          = 7.5;
      const double interface_width = 1.0;
      const bool   is_accumulative = false;

      // Use initial solution for the 2 order parameters for creating the mesh
      // which can then be reused for the cases with different maximum numbers
      // of order parameters
      Sintering::InitialValuesHypercube<dim> initial_solution(
        radius, interface_width, n_grains_dir, 2, is_accumulative);

      // Domain geometry
      auto         boundaries      = initial_solution.get_domain_boundaries();
      const double boundary_factor = 0.5;

      for (unsigned int i = 0; i < dim; i++)
        {
          boundaries.first[i] -= boundary_factor * radius;
          boundaries.second[i] += boundary_factor * radius;
        }

      // Mesh parameters
      const double       divisions_per_interface = 2;
      const bool         periodic                = false;
      const auto         global_refine           = InitialRefine::Base;
      const unsigned int max_prime               = 20;
      const double       max_level0_divisions_per_interface = 1.0 - 1e-9;
      const unsigned int divisions_per_element              = 1;

      const unsigned int n_refinements_remaining =
        create_mesh(tria,
                    boundaries.first,
                    boundaries.second,
                    interface_width,
                    divisions_per_interface,
                    periodic,
                    global_refine,
                    max_prime,
                    max_level0_divisions_per_interface,
                    divisions_per_element);

      // AMR settings
      const double       top_fraction_of_cells    = 0.9;
      const double       bottom_fraction_of_cells = 0.1;
      const unsigned int min_refinement_depth     = 3;
      const unsigned int max_refinement_depth     = 1;

      const unsigned int n_global_levels_0 =
        tria.n_global_levels() + n_refinements_remaining;

      // and limit the number of levels
      const unsigned int max_allowed_level =
        (n_global_levels_0 - 1) + max_refinement_depth;
      const unsigned int min_allowed_level =
        (n_global_levels_0 - 1) -
        std::min((n_global_levels_0 - 1), min_refinement_depth);

      // Solution vector - skip the first two components
      const unsigned int n_skip = 2;
      const unsigned int n_order_parameters =
        initial_solution.n_components() - n_skip;
      BlockVectorType solution(n_order_parameters);

      // Initializer
      const auto initialize = [&]() {
        dof_handler.distribute_dofs(fe);

        const auto relevant_dofs =
          DoFTools::extract_locally_relevant_dofs(dof_handler);

        constraints.clear();
        constraints.reinit(relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler.locally_owned_dofs(),
          relevant_dofs,
          dof_handler.get_communicator());

        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          solution.block(b).reinit(partitioner);
      };

      // Set initial conditions
      const auto set_initial_conditions =
        [&](InitialValues<dim> &initial_values) {
          for (unsigned int b = 0; b < solution.n_blocks(); ++b)
            if (b + n_skip < initial_values.n_components())
              {
                initial_values.set_component(b + n_skip);

                VectorTools::interpolate(mapping,
                                         dof_handler,
                                         initial_values,
                                         solution.block(b));

                constraints.distribute(solution.block(b));

                if (solution.block(b).has_ghost_elements())
                  solution.block(b).zero_out_ghost_values();
              }
        };

      // Make the very first initialization
      initialize();
      set_initial_conditions(initial_solution);

      // Refine mesh
      const unsigned int n_init_refinements =
        std::max(std::min(tria.n_global_levels() - 1, min_refinement_depth),
                 n_global_levels_0 - tria.n_global_levels() +
                   max_refinement_depth);

      for (unsigned int i = 0; i < n_init_refinements; ++i)
        {
          coarsen_and_refine_mesh(solution,
                                  tria,
                                  dof_handler,
                                  QGauss<dim - 1>(fe_degree + 1),
                                  top_fraction_of_cells,
                                  bottom_fraction_of_cells,
                                  min_allowed_level,
                                  max_allowed_level);

          initialize();
          set_initial_conditions(initial_solution);
        }

      // Grain tracker settings
      const bool         greedy_init              = false;
      const bool         allow_new_grains         = false;
      const bool         fast_reassignment        = false;
      const unsigned int max_order_parameters_num = n_order_parameters * 2;
      const double       threshold_lower          = 1e-15;
      const double       threshold_upper          = 1.01;
      const double       buffer_distance_ratio    = 0.05;
      const double       buffer_distance_fixed    = 0.0;
      const unsigned int order_parameters_offset  = 0;

      GrainTracker::Tracker<dim, Number> grain_tracker(dof_handler,
                                                       tria,
                                                       greedy_init,
                                                       allow_new_grains,
                                                       fast_reassignment,
                                                       max_order_parameters_num,
                                                       threshold_lower,
                                                       threshold_upper,
                                                       buffer_distance_ratio,
                                                       buffer_distance_fixed,
                                                       order_parameters_offset);

      // Analize for different number of order parameters
      for (unsigned int n_max_op_for_ic = 2;
           n_max_op_for_ic <= n_grains_per_row;
           ++n_max_op_for_ic)
        {
          // Initial distribution of particles
          Sintering::InitialValuesHypercube<dim> initial_solution_max_op(
            radius,
            interface_width,
            n_grains_dir,
            n_max_op_for_ic,
            is_accumulative);

          // Apply it
          set_initial_conditions(initial_solution_max_op);

          // Output basic info
          table.add_value("dim", dim);
          table.add_value("n_grains_per_row", n_grains_per_row);
          table.add_value("n_max_op_for_ic", n_max_op_for_ic);
          table.add_value("n_dofs", dof_handler.n_dofs());
          table.add_value("n_grains", n_grains);

          // Run benchmarks
          if constexpr (test_detect_grains)
            {
              const bool skip_reassignment = true;

              const auto time = run_measurement([&]() {
                grain_tracker.initial_setup(solution,
                                            n_order_parameters,
                                            skip_reassignment);
              });

              table.add_value("t_detect_grains", time);
              table.set_scientific("t_detect_grains", true);
            }

          if constexpr (test_initial_setup)
            {
              const bool skip_reassignment = false;

              const auto time = run_measurement([&]() {
                grain_tracker.initial_setup(solution,
                                            n_order_parameters,
                                            skip_reassignment);
              });

              table.add_value("t_initial_setup", time);
              table.set_scientific("t_initial_setup", true);
            }

          if constexpr (test_track)
            {
              const bool skip_reassignment_setup = true;
              const bool skip_reassignment_track = false;
              grain_tracker.initial_setup(solution,
                                          n_order_parameters,
                                          skip_reassignment_setup);

              const auto time = run_measurement([&]() {
                grain_tracker.track(solution,
                                    n_order_parameters,
                                    skip_reassignment_track);
              });

              table.add_value("t_track", time);
              table.set_scientific("t_track", true);
            }

          if constexpr (test_remap_dependent)
            {
              const bool skip_reassignment = true;
              grain_tracker.initial_setup(solution,
                                          n_order_parameters,
                                          skip_reassignment);

              solution.reinit(solution.n_blocks() + 1);

              std::function<void(
                std::map<unsigned int, GrainTracker::Grain<dim>> &)>
                shift_order_params =
                  [&](
                    std::map<unsigned int, GrainTracker::Grain<dim>> &grains) {
                    for (auto &[grain_id, grain] : grains)
                      {
                        const unsigned int new_op =
                          grain.get_order_parameter_id() + 1;
                        grain.set_order_parameter_id(new_op);
                      }
                  };

              grain_tracker.custom_reassignment(shift_order_params);

              const auto time =
                run_measurement([&]() { grain_tracker.remap(solution); });

              table.add_value("t_remap_dependent", time);
              table.set_scientific("t_remap_dependent", true);
            }

          if constexpr (test_remap_cycle)
            {
              set_initial_conditions(initial_solution_max_op);

              const bool skip_reassignment = true;
              grain_tracker.initial_setup(solution,
                                          n_order_parameters,
                                          skip_reassignment);

              std::function<void(
                std::map<unsigned int, GrainTracker::Grain<dim>> &)>
                shift_order_params =
                  [&](
                    std::map<unsigned int, GrainTracker::Grain<dim>> &grains) {
                    for (auto &[grain_id, grain] : grains)
                      {
                        const unsigned int new_op =
                          (grain.get_order_parameter_id() + 1) %
                          n_max_op_for_ic;
                        grain.set_order_parameter_id(new_op);
                      }
                  };

              grain_tracker.custom_reassignment(shift_order_params);

              const auto time =
                run_measurement([&]() { grain_tracker.remap(solution); });

              table.add_value("t_remap_cycle", time);
              table.set_scientific("t_remap_cycle", true);
            }
        }
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);
}
