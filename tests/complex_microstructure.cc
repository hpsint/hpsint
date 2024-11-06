// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#define MAX_SINTERING_GRAINS 20

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/sintering/initial_values_microstructure_imaging.h>
#include <pf-applications/sintering/postprocessors.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>
#include <pf-applications/grain_tracker/ellipsoid.h>
#include <pf-applications/grain_tracker/tracker.h>

#include <filesystem>
#include <iostream>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using namespace dealii;
using namespace GrainTracker;
using namespace Sintering;

using BlockVectorType = LinearAlgebra::distributed::DynamicBlockVector<double>;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm comm = MPI_COMM_WORLD;
  const bool     is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream pcout(std::cout, is_zero_rank);

  // Open cloud file
  const std::filesystem::path source_path = XSTRING(SOURCE_CODE_ROOT);
  const std::filesystem::path file_micro =
    source_path / "microstructures/81grains.micro";

  std::ifstream fstream(file_micro);
  AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

  constexpr double             interface_width = 0.4;
  constexpr InterfaceDirection interface_direction =
    InterfaceDirection::outside;

  InitialValuesMicrostructureImaging initial_values(fstream,
                                                    interface_width,
                                                    interface_direction);

  // Now build the mesh
  const unsigned int dim = 2;

  const unsigned int fe_degree      = 1;
  const unsigned int n_points_1D    = 2;
  const unsigned int n_subdivisions = 1;

  FE_Q<dim>      fe{fe_degree};
  MappingQ1<dim> mapping;
  Quadrature<1>  quad(QIterated<1>(QGauss<1>(n_points_1D), n_subdivisions));

  parallel::distributed::Triangulation<dim> tria(comm);

  const auto boundaries = initial_values.get_domain_boundaries();

  pcout << "n_grains    = " << initial_values.n_particles() << std::endl;
  pcout << "n_ops       = " << initial_values.n_order_parameters() << std::endl;
  pcout << "bottom_left = " << boundaries.first << std::endl;
  pcout << "top_right   = " << boundaries.second << std::endl;

  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {5, 1},
                                            boundaries.first,
                                            boundaries.second);

  const unsigned int n_refines_global = 4;
  const unsigned int n_refines_local  = 3;

  const double       top_fraction_of_cells    = 0.9;
  const double       bottom_fraction_of_cells = 0.1;
  const unsigned int max_refinement_depth     = 1;
  const unsigned int min_refinement_depth     = 3;
  const double       interface_val_min        = 0.05;
  const double       interface_val_max        = 0.95;
  const unsigned int op_offset                = 2;

  tria.refine_global(n_refines_global);

  const unsigned int n_global_levels_0 =
    tria.n_global_levels() + n_refines_local;

  // and limit the number of levels
  const unsigned int max_allowed_level =
    (n_global_levels_0 - 1) + max_refinement_depth;
  const unsigned int min_allowed_level =
    (n_global_levels_0 - 1) -
    std::min((n_global_levels_0 - 1), min_refinement_depth);

  DoFHandler<dim>           dof_handler(tria);
  AffineConstraints<double> constraints;
  BlockVectorType solution(op_offset + initial_values.n_order_parameters());

  const auto initialize_dofs = [&]() {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    constraints.close();

    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());

    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
      solution.block(b).reinit(partitioner);

    solution.zero_out_ghost_values();
  };

  const auto initialize_solution = [&]() {
    initialize_dofs();
    for (unsigned int c = 0; c < initial_values.n_components(); ++c)
      {
        initial_values.set_component(c);

        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 initial_values,
                                 solution.block(c));

        constraints.distribute(solution.block(c));
      }
  };
  initialize_solution();
  solution.update_ghost_values();

  const auto solution_estimate =
    solution.create_view(op_offset, solution.n_blocks());

  for (unsigned int i = 0; i < n_refines_local; ++i)
    {
      coarsen_and_refine_mesh(*solution_estimate,
                              tria,
                              dof_handler,
                              Quadrature<dim - 1>(quad),
                              top_fraction_of_cells,
                              bottom_fraction_of_cells,
                              min_allowed_level,
                              max_allowed_level,
                              interface_val_min,
                              interface_val_max);

      initialize_solution();
      solution.update_ghost_values();
    }

  // Output to vtk
  DataOut<dim> data_out;
  for (unsigned int b = op_offset; b < solution.n_blocks(); ++b)
    data_out.add_data_vector(dof_handler,
                             solution.block(b),
                             "op_" + std::to_string(b - op_offset));

  // Grain tracker settings
  const double       threshold_lower          = 1e-15;
  const double       threshold_new_grains     = 1e-15;
  const double       buffer_distance_ratio    = 0.1;
  const double       buffer_distance_fixed    = 0.0;
  const bool         allow_new_grains         = false;
  const bool         fast_reassignment        = false;
  const bool         greedy_init              = false;
  const unsigned int max_order_parameters_num = 80;

  // Lambda to remap vectors
  auto remap_vector = [&](BlockVectorType &           vector,
                          const Tracker<dim, double> &grain_tracker,
                          bool                        grains_reassigned,
                          bool                        op_number_changed) {
    const auto new_op_number =
      grain_tracker.get_active_order_parameters().size();

    if (op_number_changed &&
        new_op_number > initial_values.n_order_parameters())
      vector.reinit(op_offset +
                    grain_tracker.get_active_order_parameters().size());

    if (grains_reassigned)
      grain_tracker.remap(vector);
  };

  // Lambda to run tests
  auto test_tracking = [&](GrainRepresentation grain_representation,
                           std::string         label_representation,
                           std::string         output_prefix) {
    Tracker<dim, double> grain_tracker(dof_handler,
                                       tria,
                                       greedy_init,
                                       allow_new_grains,
                                       fast_reassignment,
                                       max_order_parameters_num,
                                       grain_representation,
                                       threshold_lower,
                                       threshold_new_grains,
                                       buffer_distance_ratio,
                                       buffer_distance_fixed,
                                       op_offset);

    const auto [n_collisions, grains_reassigned, op_number_changed] =
      grain_tracker.initial_setup(solution,
                                  initial_values.n_order_parameters());

    pcout << std::boolalpha;

    pcout << std::endl;
    pcout << label_representation << " grain representation:" << std::endl;
    pcout << "n_grains_from_gt  = " << grain_tracker.get_grains().size()
          << std::endl;
    pcout << "n_collisions      = " << n_collisions << std::endl;
    pcout << "grains_reassigned = " << grains_reassigned << std::endl;
    pcout << "op_number_changed = " << op_number_changed << std::endl;
    pcout << "new_op_number     = "
          << grain_tracker.get_active_order_parameters().size() << std::endl;

    // Remap grains to verify that we did not break the microstructure
    BlockVectorType solution_remap(solution);
    solution_remap.update_ghost_values();

    remap_vector(solution_remap,
                 grain_tracker,
                 grains_reassigned,
                 op_number_changed);

    for (unsigned int b = op_offset; b < solution_remap.n_blocks(); ++b)
      data_out.add_data_vector(dof_handler,
                               solution_remap.block(b),
                               output_prefix + "_op_" +
                                 std::to_string(b - op_offset));

    // Output tex
    const unsigned int n_op =
      grain_tracker.get_active_order_parameters().size();
    grain_tracker.track(solution_remap, n_op, true);

    const std::string filename = "81grains_" + label_representation + ".txt";

    Postprocessors::output_grain_contours(
      mapping, dof_handler, solution_remap, 0.5, filename, n_op, grain_tracker);
  };

  // Test various grain tracking representations
  test_tracking(GrainRepresentation::spherical, "Spherical", "sp");
  test_tracking(GrainRepresentation::elliptical, "Elliptical", "el");
  test_tracking(GrainRepresentation::wavefront, "Wavefront", "wf");

  // Generate output
  data_out.build_patches(mapping);
  data_out.write_vtu_in_parallel("solution_81grains.vtu", comm);
}
