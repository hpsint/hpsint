// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/trilinos/nox.h>

#include <pf-applications/base/debug.h>
#include <pf-applications/base/fe_integrator.h>
#include <pf-applications/base/scoped_name.h>
#include <pf-applications/base/solution_serialization.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/lac/evaluation.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/data_out.h>
#include <pf-applications/numerics/output.h>
#include <pf-applications/numerics/phasefield_tools.h>
#include <pf-applications/numerics/vector_tools.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/boundary_conditions.h>
#include <pf-applications/sintering/initial_values.h>
#include <pf-applications/sintering/operator_advection.h>
#include <pf-applications/sintering/operator_postproc.h>
#include <pf-applications/sintering/operator_sintering_coupled_base.h>
#include <pf-applications/sintering/output.h>
#include <pf-applications/sintering/parameters.h>
#include <pf-applications/sintering/postprocessors.h>
#include <pf-applications/sintering/preconditioners.h>
#include <pf-applications/sintering/projection.h>
#include <pf-applications/sintering/residual_wrapper.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/constraint_helper.h>
#include <pf-applications/matrix_free/output.h>
#include <pf-applications/time_integration/time_marching.h>
#include <pf-applications/time_integration/time_schemes.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim,
            template <int, typename, typename>
            typename NonLinearOperatorTpl,
            template <typename>
            typename FreeEnergyTpl,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class Problem
  {
  public:
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    // Build up sintering operator
    using NonLinearOperator =
      NonLinearOperatorTpl<dim, Number, VectorizedArrayType>;

    const Parameters                          params;
    ConditionalOStream                        pcout;
    ConditionalOStream                        pcout_statistics;
    parallel::distributed::Triangulation<dim> tria;
    std::shared_ptr<const FiniteElement<dim>> fe;
    MappingQ<dim>                             mapping;
    Quadrature<1>                             quad;
    DoFHandler<dim>                           dof_handler;

    std::unique_ptr<dealii::parallel::Helper<dim>> helper;

    AffineConstraints<Number> constraints;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    // multigrid
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;

    MGLevelObject<DoFHandler<dim>>           mg_dof_handlers;
    MGLevelObject<AffineConstraints<Number>> mg_constraints;
    MGLevelObject<MGTwoLevelTransfer<dim, typename VectorType::BlockType>>
                                                                transfers;
    MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>> mg_matrix_free;

    std::shared_ptr<
      MGTransferGlobalCoarsening<dim, typename VectorType::BlockType>>
      transfer;


    std::pair<dealii::Point<dim>, dealii::Point<dim>>
           geometry_domain_boundaries;
    double geometry_r_max;
    double geometry_interface_width;

    unsigned int n_global_levels_0;
    unsigned int current_max_refinement_depth;
    double       current_min_mesh_quality;
    double       time_last_output;
    unsigned int n_timestep;
    unsigned int n_timestep_last_amr;
    unsigned int n_timestep_last_gt;
    unsigned int n_linear_iterations;
    unsigned int n_non_linear_iterations;
    unsigned int n_residual_evaluations;
    unsigned int n_failed_tries;
    unsigned int n_failed_linear_iterations;
    unsigned int n_failed_non_linear_iterations;
    unsigned int n_failed_residual_evaluations;
    double       max_reached_dt;
    unsigned int restart_counter;
    double       t;

    std::map<std::string, unsigned int> counters;

    Problem(const Parameters                   &params,
            std::shared_ptr<InitialValues<dim>> initial_solution,
            std::ostream                       &out,
            std::ostream                       &out_statistics)
      : params(params)
      , pcout(out,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(out_statistics,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , mapping(1)
      , quad(QIterated<1>(QGauss<1>(params.approximation_data.n_points_1D),
                          params.approximation_data.n_subdivisions))
      , dof_handler(tria)
    {
      MyScope("Problem::constructor");

      fe = create_fe(params.approximation_data.fe_degree,
                     params.approximation_data.n_subdivisions);

      geometry_domain_boundaries = initial_solution->get_domain_boundaries();
      geometry_r_max             = initial_solution->get_r_max();
      geometry_interface_width   = initial_solution->get_interface_width();
      time_last_output           = 0;
      n_timestep                 = 0;
      n_timestep_last_amr        = 0;
      n_timestep_last_gt         = 0;
      n_linear_iterations        = 0;
      n_non_linear_iterations    = 0;
      n_residual_evaluations     = 0;
      n_failed_tries             = 0;
      n_failed_linear_iterations = 0;
      n_failed_non_linear_iterations = 0;
      n_failed_residual_evaluations  = 0;
      max_reached_dt                 = 0.0;
      restart_counter                = 0;
      counters                       = {};

      // Current time
      t = params.time_integration_data.time_start;

      // Initialize timestepping
      auto scheme_variant = TimeIntegration::create_time_scheme<Number>(
        params.time_integration_data.integration_scheme);

      TimeIntegration::TimeIntegratorData<Number> time_data(
        scheme_variant.template try_take<ImplicitScheme<Number>>(),
        params.time_integration_data.time_step_init);

      // Parse initial refinement options
      InitialRefine global_refine;
      if (params.geometry_data.global_refinement == "None")
        global_refine = InitialRefine::None;
      else if (params.geometry_data.global_refinement == "Base")
        global_refine = InitialRefine::Base;
      else if (params.geometry_data.global_refinement == "Full")
        global_refine = InitialRefine::Full;
      else
        AssertThrow(false, ExcNotImplemented());

      // Parse initial mesh otpions
      InitialMesh initial_mesh;
      if (params.geometry_data.initial_mesh == "Interface")
        initial_mesh = InitialMesh::Interface;
      else if (params.geometry_data.initial_mesh == "MaxRadius")
        initial_mesh = InitialMesh::MaxRadius;
      else if (params.geometry_data.initial_mesh == "Divisions")
        initial_mesh = InitialMesh::Divisions;
      else
        AssertThrow(false, ExcNotImplemented());

      const unsigned int n_refinements_remaining =
        create_grid(initial_mesh, global_refine);
      n_global_levels_0 = tria.n_global_levels() + n_refinements_remaining;

      // Set the maximum refinement depth
      current_max_refinement_depth =
        params.adaptivity_data.max_refinement_depth;

      // Set the minimum mesh quality
      current_min_mesh_quality = params.adaptivity_data.quality_min;

      initialize();

      const auto initialize_solution =
        [&](std::vector<typename VectorType::BlockType *> solution_ptr,
            MyTimerOutput                                &timer) {
          ScopedName sc("initialize_solution");
          MyScope    scope(timer, sc);

          AssertThrow(initial_solution->n_components() <= solution_ptr.size(),
                      ExcMessage(
                        "Number of initial values components (" +
                        std::to_string(initial_solution->n_components()) +
                        ") exceeds the number of blocks provided (" +
                        std::to_string(solution_ptr.size()) + ")."));

          for (unsigned int c = 0; c < initial_solution->n_components(); ++c)
            {
              initial_solution->set_component(c);

              VectorTools::interpolate(mapping,
                                       dof_handler,
                                       *initial_solution,
                                       *solution_ptr[c]);

              constraints.distribute(*solution_ptr[c]);
              solution_ptr[c]->zero_out_ghost_values();
            }
        };

      const auto initialize_grain_tracker =
        [](GrainTracker::Tracker<dim, Number> &grain_tracker,
           std::function<void(bool)>           run_gt_callback) {
          (void)grain_tracker;
          run_gt_callback(true);
        };

      run(initial_solution->n_components(),
          time_data,
          initialize_solution,
          initialize_grain_tracker);
    }

    Problem(const Parameters  &params,
            const std::string &restart_path,
            std::ostream      &out,
            std::ostream      &out_statistics)
      : params(params)
      , pcout(out,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(out_statistics,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , mapping(1)
      , quad(QIterated<1>(QGauss<1>(params.approximation_data.n_points_1D),
                          params.approximation_data.n_subdivisions))
      , dof_handler(tria)
    {
      MyScope("Problem::constructor");

      fe = create_fe(params.approximation_data.fe_degree,
                     params.approximation_data.n_subdivisions);

      // 0) load internal state
      unsigned int n_ranks;
      unsigned int n_initial_components;
      unsigned int n_blocks_per_vector;
      unsigned int n_blocks_total;
      bool         flexible_output;
      bool         full_history;

      std::ifstream                   in_stream(restart_path + "_driver");
      boost::archive::binary_iarchive fisb(in_stream);
      fisb >> flexible_output;
      fisb >> full_history;
      fisb >> n_ranks;

      const auto n_mpi_processes =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

      AssertThrow(
        flexible_output || (n_ranks == n_mpi_processes),
        ExcMessage(
          "You are not using flexible serialization. You can restart this simulation only with " +
          std::to_string(n_ranks) + "! But you have " +
          std::to_string(n_mpi_processes) + "!"));

      fisb >> n_initial_components;
      fisb >> n_blocks_per_vector;
      fisb >> n_blocks_total;

      auto scheme_variant = TimeIntegration::create_time_scheme<Number>(
        params.time_integration_data.integration_scheme);

      // Read the rest
      fisb >> *this;

      // Load grains
      std::ifstream                   in_stream_gt(restart_path + "_grains");
      boost::archive::binary_iarchive figt(in_stream_gt);
      std::vector<GrainTracker::Grain<dim>> grains;
      figt >> grains;

      // Load time data
      std::ifstream                   in_stream_time(restart_path + "_time");
      boost::archive::binary_iarchive fitime(in_stream_time);
      TimeIntegration::TimeIntegratorData<Number> time_data;
      fitime >> time_data;

      // Check if the data structures are consistent
      if (full_history)
        {
          // Strictly speaking, the number of vectors should be equal to the
          // time integration order + 1, however, we skipped the recent old
          // solution since it will get overwritten anyway, so we neither save
          // it not load during restarts.
          AssertDimension(time_data.get_order(),
                          n_blocks_total / n_blocks_per_vector);

          // We reinit time data anyways since the user might have changed the
          // integration scheme beetwen runs
          time_data = TimeIntegration::TimeIntegratorData<Number>(
            time_data,
            scheme_variant.template try_take<ImplicitScheme<Number>>());
        }
      else
        {
          // We reinit time data anyways since we do not contain enough history
          // vectors and will start from the lowest order integration scheme
          time_data = TimeIntegration::TimeIntegratorData<Number>(
            scheme_variant.template try_take<ImplicitScheme<Number>>(),
            time_data.get_current_dt());
        }

      // Parse initial mesh otpions
      InitialMesh initial_mesh;
      if (params.geometry_data.initial_mesh == "Interface")
        initial_mesh = InitialMesh::Interface;
      else if (params.geometry_data.initial_mesh == "MaxRadius")
        initial_mesh = InitialMesh::MaxRadius;
      else if (params.geometry_data.initial_mesh == "Divisions")
        initial_mesh = InitialMesh::Divisions;
      else
        AssertThrow(false, ExcNotImplemented());

      pcout << "Loading restart data at t = " << t << " (" << restart_path
            << ")" << std::endl;

      // 1) create coarse mesh
      create_grid(initial_mesh, InitialRefine::None);

      // 2) load mesh refinement (incl. vectors)
      tria.load(restart_path + "_tria");

      // note: for flexible restart file, do not repartition here,
      // since else the attached vectors will be lost; in the case of
      // non-flexible restart filed, we need to repartition here, since
      // Triangulation::load() does not guarantee that the saved and loaded
      // files are identical
      if (params.restart_data.flexible_output == false)
        tria.repartition();

      // 3) initialize data structures
      initialize();

      // 4) helper function to initialize solution vector
      const auto initialize_solution =
        [&, flexible_output, n_blocks_total, restart_path](
          std::vector<typename VectorType::BlockType *> solution_ptr,
          MyTimerOutput                                &timer) {
          ScopedName sc("deserialize_solution");
          MyScope    scope(timer, sc);

          if (n_blocks_total < solution_ptr.size())
            solution_ptr.resize(n_blocks_total);

          if (flexible_output)
            {
              parallel::distributed::
                SolutionTransfer<dim, typename VectorType::BlockType>
                  solution_transfer(dof_handler);

              // In order to perform deserialization, we need to add dummy
              // vectors which will get deleted at the end
              std::vector<std::shared_ptr<typename VectorType::BlockType>>
                dummy_vecs;
              if (n_blocks_total > solution_ptr.size())
                for (unsigned int i = solution_ptr.size(); i < n_blocks_total;
                     ++i)
                  {
                    auto dummy =
                      std::make_shared<typename VectorType::BlockType>(
                        *solution_ptr[0]);
                    dummy_vecs.push_back(dummy);
                    solution_ptr.push_back(dummy.get());
                  }

              solution_transfer.deserialize(solution_ptr);
            }
          else
            {
              parallel::distributed::
                SolutionSerialization<dim, typename VectorType::BlockType>
                  solution_serialization(dof_handler);

              solution_serialization.add_vectors(solution_ptr);

              solution_serialization.load(restart_path + "_vectors");
            }
        };

      // 5) load grains to grain tracker
      const auto initialize_grain_tracker =
        [grains =
           std::move(grains)](GrainTracker::Tracker<dim, Number> &grain_tracker,
                              std::function<void(bool)> run_gt_callback) {
          grain_tracker.load_grains(grains.cbegin(), grains.cend());
          run_gt_callback(false);
        };

      // 6) run time loop
      run(n_initial_components,
          time_data,
          initialize_solution,
          initialize_grain_tracker);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      // geometry
      ar &geometry_domain_boundaries;
      ar &geometry_r_max;
      ar &geometry_interface_width;

      // counters
      ar &n_global_levels_0;
      ar &current_max_refinement_depth;
      ar &current_min_mesh_quality;
      ar &time_last_output;
      ar &n_timestep;
      ar &n_timestep_last_amr;
      ar &n_timestep_last_gt;
      ar &n_linear_iterations;
      ar &n_non_linear_iterations;
      ar &n_residual_evaluations;
      ar &n_failed_tries;
      ar &n_failed_linear_iterations;
      ar &n_failed_non_linear_iterations;
      ar &n_failed_residual_evaluations;
      ar &max_reached_dt;
      ar &restart_counter;
      ar &t;
      ar &counters;
    }

    unsigned int
    create_grid(InitialMesh initial_mesh, InitialRefine global_refine)
    {
      MyScope("Problem::create_grid");

      std::pair<dealii::Point<dim>, dealii::Point<dim>> boundaries;

      if (!params.geometry_data.custom_bounding_box)
        {
          boundaries  = geometry_domain_boundaries;
          double rmax = geometry_r_max;

          for (unsigned int i = 0; i < dim; i++)
            {
              boundaries.first[i] -=
                params.geometry_data.boundary_factor * rmax;
              boundaries.second[i] +=
                params.geometry_data.boundary_factor * rmax;
            }
        }
      else
        {
          if constexpr (dim >= 1)
            {
              boundaries.first[0] =
                params.geometry_data.bounding_box_data.x_min;
              boundaries.second[0] =
                params.geometry_data.bounding_box_data.x_max;
            }
          if constexpr (dim >= 2)
            {
              boundaries.first[1] =
                params.geometry_data.bounding_box_data.y_min;
              boundaries.second[1] =
                params.geometry_data.bounding_box_data.y_max;
            }
          if constexpr (dim == 3)
            {
              boundaries.first[2] =
                params.geometry_data.bounding_box_data.z_min;
              boundaries.second[2] =
                params.geometry_data.bounding_box_data.z_max;
            }
        }

      unsigned int n_refinements_remaining = 0;
      if (initial_mesh == InitialMesh::Interface)
        {
          n_refinements_remaining = create_mesh_from_interface(
            tria,
            boundaries.first,
            boundaries.second,
            geometry_interface_width,
            params.geometry_data.divisions_per_interface,
            params.geometry_data.periodic,
            global_refine,
            params.geometry_data.max_prime,
            params.geometry_data.max_level0_divisions_per_interface,
            params.approximation_data.n_subdivisions,
            params.print_time_loop);
        }
      else if (initial_mesh == InitialMesh::MaxRadius)
        {
          n_refinements_remaining = create_mesh_from_radius(
            tria,
            boundaries.first,
            boundaries.second,
            geometry_interface_width,
            params.geometry_data.divisions_per_interface,
            geometry_r_max,
            params.geometry_data.periodic,
            global_refine,
            params.geometry_data.max_prime,
            params.approximation_data.n_subdivisions,
            params.print_time_loop);
        }
      else
        {
          std::vector<unsigned int> subdivisions(dim);

          if constexpr (dim >= 1)
            subdivisions[0] = params.geometry_data.divisions_data.nx;
          if constexpr (dim >= 2)
            subdivisions[1] = params.geometry_data.divisions_data.ny;
          if constexpr (dim >= 3)
            subdivisions[2] = params.geometry_data.divisions_data.nz;

          n_refinements_remaining =
            create_mesh_from_divisions(tria,
                                       boundaries.first,
                                       boundaries.second,
                                       subdivisions,
                                       params.geometry_data.periodic,
                                       0);
        }

      helper = std::make_unique<dealii::parallel::Helper<dim>>(tria);

      const auto weight_function = parallel::hanging_nodes_weighting<dim>(
        *helper, params.geometry_data.hanging_node_weight);
      tria.signals.weight.connect(weight_function);

      tria.repartition();

      return n_refinements_remaining;
    }

    void
    initialize(const unsigned int n_components = 0)
    {
      // setup DoFHandlers, ...
      if (true)
        {
          MyScope("Problem::initialize::dofhandler");
          dof_handler.distribute_dofs(*fe);
        }

      // ... constraints, and ...
      if (true)
        {
          MyScope("Problem::initialize::constraints");
          constraints.clear();
          constraints.reinit(
            DoFTools::extract_locally_relevant_dofs(dof_handler));
          DoFTools::make_hanging_node_constraints(dof_handler, constraints);

          if (params.geometry_data.periodic)
            {
              std::vector<GridTools::PeriodicFacePair<
                typename dealii::DoFHandler<dim>::cell_iterator>>
                periodicity_vector;

              for (unsigned int d = 0; d < dim; ++d)
                {
                  GridTools::collect_periodic_faces(
                    dof_handler, 2 * d, 2 * d + 1, d, periodicity_vector);
                }

              DoFTools::make_periodicity_constraints<dim, dim>(
                periodicity_vector, constraints);
            }

          constraints.close();
        }

      // ... MatrixFree
      if (true)
        {
          MyScope("Problem::initialize::matrix_free");
          typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
            additional_data;
          additional_data.mapping_update_flags =
            update_values | update_gradients;

          if (params.advection_data.enable ||
              (params.output_data.table &&
               !params.output_data.domain_integrals.empty() &&
               (!params.output_data.control_boxes.empty() ||
                params.output_data.auto_control_box)))
            additional_data.mapping_update_flags |= update_quadrature_points;

          additional_data.allow_ghosted_vectors_in_loops    = false;
          additional_data.overlap_communication_computation = false;

          matrix_free.reinit(
            mapping, dof_handler, constraints, quad, additional_data);
        }

      if ((params.preconditioners_data.outer_preconditioner == "GMG") ||
          (params.preconditioners_data.outer_preconditioner == "BlockGMG") ||
          ((params.preconditioners_data.outer_preconditioner ==
            "BlockPreconditioner2") &&
           ((params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "GMG") ||
            (params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "BlockGMG"))))
        {
          MyScope("Problem::initialize::multigrid");

          mg_triangulations = MGTransferGlobalCoarseningTools::
            create_geometric_coarsening_sequence(tria);

          const unsigned int min_level = 0;
          const unsigned int max_level = mg_triangulations.size() - 1;

          const unsigned int max_max_level =
            n_global_levels_0 + params.adaptivity_data.max_refinement_depth;

          mg_dof_handlers.resize(min_level, max_level);
          transfers.resize(min_level, max_level);

          if (min_level != mg_constraints.min_level() ||
              max_max_level != mg_constraints.max_level())
            mg_constraints.resize(min_level, max_max_level);

          if (min_level != mg_matrix_free.min_level() ||
              max_max_level != mg_matrix_free.max_level())
            mg_matrix_free.resize(min_level, max_max_level);

          for (unsigned int l = min_level; l <= max_level; ++l)
            {
              auto &dof_handler = mg_dof_handlers[l];
              auto &constraints = mg_constraints[l];
              auto &matrix_free = mg_matrix_free[l];

              dof_handler.reinit(*mg_triangulations[l]);
              dof_handler.distribute_dofs(*fe);

              constraints.clear();
              constraints.reinit(
                DoFTools::extract_locally_relevant_dofs(dof_handler));
              DoFTools::make_hanging_node_constraints(dof_handler, constraints);
              Assert(params.geometry_data.periodic == false,
                     ExcNotImplemented());
              constraints.close();

              typename MatrixFree<dim, Number, VectorizedArrayType>::
                AdditionalData additional_data;
              additional_data.mapping_update_flags =
                update_values | update_gradients;

              additional_data.allow_ghosted_vectors_in_loops    = false;
              additional_data.overlap_communication_computation = false;

              matrix_free.reinit(
                mapping, dof_handler, constraints, quad, additional_data);
            }

          for (auto l = min_level; l < max_level; ++l)
            transfers[l + 1].reinit(mg_dof_handlers[l + 1],
                                    mg_dof_handlers[l],
                                    mg_constraints[l + 1],
                                    mg_constraints[l]);

          transfer = std::make_shared<
            MGTransferGlobalCoarsening<dim, typename VectorType::BlockType>>(
            transfers, [&](const auto l, auto &vec) {
              mg_matrix_free[l].initialize_dof_vector(vec);
            });
        }

      types::global_cell_index n_cells_w_hn  = 0;
      types::global_cell_index n_cells_wo_hn = 0;

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (helper->is_constrained(cell))
              n_cells_w_hn++;
            else
              n_cells_wo_hn++;
          }

      const auto min_max_avg_n_cells_w_hn =
        Utilities::MPI::min_max_avg(n_cells_w_hn, MPI_COMM_WORLD);
      const auto min_max_avg_n_cells_wo_hn =
        Utilities::MPI::min_max_avg(n_cells_wo_hn, MPI_COMM_WORLD);
      const auto min_max_avg_n_cells =
        Utilities::MPI::min_max_avg(n_cells_w_hn + n_cells_wo_hn,
                                    MPI_COMM_WORLD);

      AssertDimension(
        static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.sum) +
          static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.sum),
        tria.n_global_active_cells());

      std::vector<unsigned int> unrefined_cells(tria.n_active_cells(), false);

      for (const auto &cell : tria.cell_iterators())
        if (cell->has_children())
          {
            bool flag = true;

            for (unsigned int c = 0; c < cell->n_children(); ++c)
              if (cell->child(c)->is_active() == false ||
                  cell->child(c)->is_locally_owned() == false)
                flag = false;

            if (flag)
              for (unsigned int c = 0; c < cell->n_children(); ++c)
                unrefined_cells[cell->child(c)->active_cell_index()] = true;
          }


      unsigned int n_unrefined_cells = 0;
      for (const auto i : unrefined_cells)
        if (i)
          n_unrefined_cells++;

      const auto min_max_avg_n_unrefined_cells =
        Utilities::MPI::min_max_avg(n_unrefined_cells, MPI_COMM_WORLD);


      // clang-format off
      pcout_statistics << "System statistics:" << std::endl;
      pcout_statistics << "  - n cell:                    " << static_cast<types::global_cell_index>(min_max_avg_n_cells.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n cell w hanging nodes:    " << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n cell wo hanging nodes:   " << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n cell fine batch:         " << static_cast<types::global_cell_index>(min_max_avg_n_unrefined_cells.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_unrefined_cells.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_unrefined_cells.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;
      pcout_statistics << "  - n coarse cells:            " << tria.n_cells(0) << std::endl;
      pcout_statistics << "  - n theoretical fine cells:  " << tria.n_cells(0) * Utilities::pow<std::int64_t>(2*dim, tria.n_global_levels() - 1) << std::endl;

      if(transfer)
      {
      pcout_statistics << "  - n cells on levels:         ";
      for(const auto & tria : mg_triangulations)
        pcout_statistics << tria->n_global_active_cells () << " ";
      pcout_statistics << std::endl;
      }

      pcout_statistics << "  - n dofs:                    " << dof_handler.n_dofs() << std::endl;
      if(n_components > 0)
        pcout_statistics << "  - n components:              " << n_components << std::endl;
      pcout_statistics << std::endl;
      // clang-format on
    }

    void
    run(const unsigned int                          n_initial_components,
        TimeIntegration::TimeIntegratorData<Number> time_data,
        const std::function<void(std::vector<typename VectorType::BlockType *>,
                                 MyTimerOutput &)> &initialize_solution,
        const std::function<void(GrainTracker::Tracker<dim, Number> &,
                                 std::function<void(bool)>)>
          &initialize_grain_tracker)
    {
      TimerPredicate restart_predicate(params.restart_data.type,
                                       params.restart_data.type == "n_calls" ?
                                         n_timestep :
                                         t,
                                       params.restart_data.interval);

      // Compute energy and mobility parameters
      double                            A, B, kappa_c, kappa_p;
      std::shared_ptr<MobilityProvider> mobility_provider;
      if (params.material_data.type == "Abstract")
        {
          A       = params.material_data.energy_abstract_data.A;
          B       = params.material_data.energy_abstract_data.B;
          kappa_c = params.material_data.energy_abstract_data.kappa_c;
          kappa_p = params.material_data.energy_abstract_data.kappa_p;

          mobility_provider = std::make_shared<ProviderAbstract>(
            params.material_data.mobility_abstract_data.Mvol,
            params.material_data.mobility_abstract_data.Mvap,
            params.material_data.mobility_abstract_data.Msurf,
            params.material_data.mobility_abstract_data.Mgb,
            params.material_data.mobility_abstract_data.L);
        }
      else
        {
          const double gamma_surf =
            params.material_data.energy_realistic_data.surface_energy;
          const double gamma_gb =
            params.material_data.energy_realistic_data.grain_boundary_energy;

          auto energy_params =
            compute_energy_params(gamma_surf,
                                  gamma_gb,
                                  geometry_interface_width,
                                  params.material_data.length_scale,
                                  params.material_data.energy_scale);

          A       = energy_params.A;
          B       = energy_params.B;
          kappa_c = energy_params.kappa_c;
          kappa_p = energy_params.kappa_p;

          const auto temperature_function =
            std::make_shared<Function1DPiecewise<double>>(
              params.material_data.temperature);

          mobility_provider = std::make_shared<ProviderRealistic>(
            params.material_data.mobility_realistic_data.omega,
            params.material_data.mobility_realistic_data.D_vol0,
            params.material_data.mobility_realistic_data.D_vap0,
            params.material_data.mobility_realistic_data.D_surf0,
            params.material_data.mobility_realistic_data.D_gb0,
            params.material_data.mobility_realistic_data.Q_vol,
            params.material_data.mobility_realistic_data.Q_vap,
            params.material_data.mobility_realistic_data.Q_surf,
            params.material_data.mobility_realistic_data.Q_gb,
            params.material_data.mobility_realistic_data.D_gb_mob0,
            params.material_data.mobility_realistic_data.Q_gb_mob,
            geometry_interface_width,
            params.material_data.time_scale,
            params.material_data.length_scale,
            params.material_data.energy_scale,
            temperature_function);

          const auto mobility_reference = mobility_provider->calculate(0.0);

          // Output material data
          pcout << "Effective model material data: " << std::endl;

          pcout << "- energy parameters:" << std::endl;
          pcout << "  A       = " << A << std::endl;
          pcout << "  B       = " << B << std::endl;
          pcout << "  kappa_c = " << kappa_c << std::endl;
          pcout << "  kappa_p = " << kappa_p << std::endl;

          pcout << "- mobility parameters:" << std::endl;
          pcout << "  Mvol    = " << mobility_reference.Mvol << std::endl;
          pcout << "  Mvap    = " << mobility_reference.Mvap << std::endl;
          pcout << "  Msurf   = " << mobility_reference.Msurf << std::endl;
          pcout << "  Mgb     = " << mobility_reference.Mgb << std::endl;
          pcout << "  L       = " << mobility_reference.L << std::endl;

          pcout << std::endl;
        }

      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        kappa_c,
        kappa_p,
        mobility_provider,
        std::move(time_data),
        FreeEnergyTpl<VectorizedArrayType>::op_components_offset);

      pcout << "Mobility type: "
            << (sintering_data.use_tensorial_mobility ? "tensorial" : "scalar")
            << std::endl;
      pcout << std::endl;

      sintering_data.set_n_components(n_initial_components);

      // Get the current timestep size
      auto dt = sintering_data.time_data.get_current_dt();

      TimeIntegration::SolutionHistory<VectorType> solution_history(
        sintering_data.time_data.get_order() + 1);

      MGLevelObject<SinteringOperatorData<dim, VectorizedArrayType>>
        mg_sintering_data(0,
                          n_global_levels_0 +
                            params.adaptivity_data.max_refinement_depth,
                          sintering_data);

      // New grains can not appear in current sintering simulations
      const unsigned int order_parameters_offset =
        FreeEnergyTpl<VectorizedArrayType>::op_components_offset;
      const bool allow_new_grains = false;
      const bool do_timing        = true;
      const bool do_logging       = params.grain_tracker_data.verbosity >= 1;

      // Grain representation type
      GrainTracker::GrainRepresentation grain_representation;
      if (params.grain_tracker_data.grain_representation == "Spherical")
        grain_representation = GrainTracker::GrainRepresentation::spherical;
      else if (params.grain_tracker_data.grain_representation == "Elliptical")
        grain_representation = GrainTracker::GrainRepresentation::elliptical;
      else if (params.grain_tracker_data.grain_representation == "Wavefront")
        grain_representation = GrainTracker::GrainRepresentation::wavefront;
      else
        AssertThrow(false, ExcNotImplemented());

      GrainTracker::Tracker<dim, Number> grain_tracker(
        dof_handler,
        tria,
        !params.geometry_data.minimize_order_parameters,
        allow_new_grains,
        params.grain_tracker_data.fast_reassignment,
        MAX_SINTERING_GRAINS,
        grain_representation,
        params.grain_tracker_data.threshold_lower,
        params.grain_tracker_data.threshold_new_grains,
        params.grain_tracker_data.buffer_distance_ratio,
        params.grain_tracker_data.buffer_distance_fixed,
        order_parameters_offset,
        do_timing,
        do_logging,
        params.grain_tracker_data.use_old_remap);

      // Advection physics for shrinkage
      AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism(
        params.advection_data.enable,
        params.advection_data.mt,
        params.advection_data.mr);

      // Advection operator
      AdvectionOperator<dim, Number, VectorizedArrayType> advection_operator(
        params.advection_data.k,
        params.advection_data.cgb,
        params.advection_data.ceq,
        params.advection_data.smoothening,
        matrix_free,
        constraints,
        sintering_data,
        grain_tracker,
        advection_mechanism);

      // Mechanics material data - plane type relevant for 2D
      Structural::MaterialPlaneType plane_type;
      if (params.material_data.mechanics_data.plane_type == "None")
        plane_type = Structural::MaterialPlaneType::none;
      else if (params.material_data.mechanics_data.plane_type == "PlaneStrain")
        plane_type = Structural::MaterialPlaneType::plane_strain;
      else if (params.material_data.mechanics_data.plane_type == "PlaneStress")
        plane_type = Structural::MaterialPlaneType::plane_stress;
      else
        AssertThrow(false, ExcNotImplemented());

      FreeEnergyTpl<VectorizedArrayType> free_energy(A, B);

      auto nonlinear_operator = NonLinearOperator::create(
        matrix_free,
        constraints,
        free_energy,
        sintering_data,
        solution_history,
        advection_mechanism,
        params.matrix_based,
        params.use_tensorial_mobility_gradient_on_the_fly,
        params.material_data.mechanics_data.E,
        params.material_data.mechanics_data.nu,
        plane_type,
        params.material_data.mechanics_data.c_min);

      // Create residual wrapper depending on whether we have advection or not
      std::unique_ptr<ResidualWrapper<Number>> residual_wrapper;

      if (params.advection_data.enable == false)
        residual_wrapper =
          std::make_unique<ResidualWrapperGeneric<Number, NonLinearOperator>>(
            nonlinear_operator);
      else
        residual_wrapper =
          std::make_unique<ResidualWrapperAdvection<dim,
                                                    Number,
                                                    VectorizedArrayType,
                                                    NonLinearOperator>>(
            advection_operator, nonlinear_operator);



      // Save all blocks at quadrature points if the advection mechanism is
      // enabled
      const bool save_all_blocks = params.advection_data.enable;

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (transfer)
        preconditioner =
          std::make_unique<BlockPreconditioner2<dim,
                                                Number,
                                                VectorizedArrayType,
                                                NonLinearOperatorTpl>>(
            nonlinear_operator,
            sintering_data,
            matrix_free,
            constraints,
            mg_sintering_data,
            mg_matrix_free,
            mg_constraints,
            transfer,
            params.preconditioners_data.block_preconditioner_2_data,
            params.print_time_loop);
      else if (params.preconditioners_data.outer_preconditioner ==
               "BlockPreconditioner2")
        {
          if constexpr (std::is_base_of_v<
                          SinteringOperatorCoupledBase<dim,
                                                       Number,
                                                       VectorizedArrayType,
                                                       NonLinearOperator>,
                          NonLinearOperator>)
            preconditioner =
              std::make_unique<BlockPreconditioner2<dim,
                                                    Number,
                                                    VectorizedArrayType,
                                                    NonLinearOperatorTpl>>(
                nonlinear_operator,
                sintering_data,
                matrix_free,
                constraints,
                params.preconditioners_data.block_preconditioner_2_data,
                advection_mechanism,
                nonlinear_operator.get_zero_constraints_indices(),
                params.material_data.mechanics_data.E,
                params.material_data.mechanics_data.nu,
                plane_type,
                params.print_time_loop);
          else
            preconditioner =
              std::make_unique<BlockPreconditioner2<dim,
                                                    Number,
                                                    VectorizedArrayType,
                                                    NonLinearOperatorTpl>>(
                nonlinear_operator,
                sintering_data,
                matrix_free,
                constraints,
                params.preconditioners_data.block_preconditioner_2_data,
                advection_mechanism,
                params.print_time_loop);
        }
      else
        preconditioner = Preconditioners::create(
          nonlinear_operator, params.preconditioners_data.outer_preconditioner);

      // A check for validity of the FDM approximation and direct linear solver
      if (params.nonlinear_data.fdm_jacobian_approximation)
        {
          AssertThrow(params.matrix_based, ExcNotImplemented());
        }
      if (params.nonlinear_data.l_solver == "Direct")
        {
          AssertThrow(
            params.matrix_based,
            ExcMessage(
              "A matrix-based mode has to be enabled to use the direct solver"));
        }

      MyTimerOutput timer(pcout.get_stream());
      TimerCollection::configure(params.profiling_data.output_time_interval);

      // Define vector to store additional initializers for additional vectors
      std::vector<std::function<void()>> additional_initializations;

      // Initialization of data for advanced postprocessing (if needed)
      PostprocOperator<dim, Number, VectorizedArrayType> postproc_operator(
        matrix_free, constraints, sintering_data, params.matrix_based);

      MassMatrix<dim, Number, VectorizedArrayType> mass_operator(matrix_free,
                                                                 constraints);

      std::unique_ptr<ReductionControl> postproc_solver_control_l;
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        postproc_preconditioner;
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>>
        postproc_linear_solver;

      VectorType postproc_lhs, postproc_rhs;

      if (params.output_data.fluxes_divergences)
        {
          postproc_solver_control_l =
            std::make_unique<ReductionControl>(params.nonlinear_data.l_max_iter,
                                               params.nonlinear_data.l_abs_tol,
                                               params.nonlinear_data.l_rel_tol);

          postproc_preconditioner =
            Preconditioners::create(mass_operator, "InverseDiagonalMatrix");

          postproc_linear_solver =
            std::make_unique<LinearSolvers::SolverGMRESWrapper<
              MassMatrix<dim, Number, VectorizedArrayType>,
              Preconditioners::PreconditionerBase<Number>>>(
              mass_operator,
              *postproc_preconditioner,
              *postproc_solver_control_l,
              params.nonlinear_data.gmres_data);

          additional_initializations.emplace_back(
            [&postproc_operator, &postproc_lhs, &postproc_rhs]() {
              postproc_operator.initialize_dof_vector(postproc_lhs);
              postproc_operator.initialize_dof_vector(postproc_rhs);
            });
        }

      // Non-linear Newton solver statistics
      NonLinearSolvers::NewtonSolverSolverControl statistics(
        params.nonlinear_data.nl_max_iter,
        params.nonlinear_data.nl_abs_tol,
        params.nonlinear_data.nl_rel_tol);

      // Lambda to set up linearization point, not const as I move it later
      auto nl_setup_linearization_point = [&](const VectorType &current_u) {
        sintering_data.fill_quadrature_point_values(
          matrix_free,
          current_u,
          params.advection_data.enable,
          save_all_blocks);
      };

      // Lambda to update preconditioner, not const as I move it later
      auto nl_setup_custom_preconditioner = [&](const VectorType &current_u) {
        if (transfer) // update multigrid levels
          {
            const unsigned int min_level = transfers.min_level();
            const unsigned int max_level = transfers.max_level();
            const unsigned int n_blocks  = current_u.n_blocks();

            for (unsigned int l = min_level; l <= max_level; ++l)
              mg_sintering_data[l].time_data.set_all_dt(
                sintering_data.time_data.get_all_dt());

            MGLevelObject<VectorType> mg_current_u(min_level, max_level);

            // acitve level
            mg_current_u[max_level].reinit(n_blocks);
            for (unsigned int b = 0; b < n_blocks; ++b)
              mg_matrix_free[max_level].initialize_dof_vector(
                mg_current_u[max_level].block(b));
            mg_current_u[max_level].copy_locally_owned_data_from(current_u);

            // coarser levels
            for (unsigned int l = max_level; l > min_level; --l)
              {
                mg_current_u[l - 1].reinit(n_blocks);
                for (unsigned int b = 0; b < n_blocks; ++b)
                  mg_matrix_free[l - 1].initialize_dof_vector(
                    mg_current_u[l - 1].block(b));

                for (unsigned int b = 0; b < n_blocks; ++b)
                  transfers[l].interpolate(mg_current_u[l - 1].block(b),
                                           mg_current_u[l].block(b));
              }

            for (unsigned int l = min_level; l <= max_level; ++l)
              {
                mg_sintering_data[l].set_n_components(
                  sintering_data.n_components());
                mg_sintering_data[l].fill_quadrature_point_values(
                  mg_matrix_free[l],
                  mg_current_u[l],
                  params.advection_data.enable,
                  save_all_blocks);
              }
          }
      };

      // Assemble quantities to check, not const as I move it later
      auto nl_quantities_to_check = [&]() {
        std::vector<std::tuple<std::string, unsigned int, unsigned int>>
          check_qtys;

        if (order_parameters_offset > 0)
          check_qtys.emplace_back("ch", 0, 1);

        if (order_parameters_offset > 1)
          check_qtys.emplace_back("mu", 1, 1);

        const unsigned n_ac =
          sintering_data.n_components() - order_parameters_offset;
        if (n_ac)
          check_qtys.emplace_back("ac", order_parameters_offset, n_ac);

        const bool has_mec =
          nonlinear_operator.n_components() > sintering_data.n_components();
        if (has_mec)
          check_qtys.emplace_back("mec",
                                  nonlinear_operator.n_components() - dim,
                                  dim);

        return check_qtys;
      };

      std::unique_ptr<TimeIntegration::TimeMarching<VectorType>> time_marching;

      // So far a single time marching option here, but later more will be added
      time_marching = std::make_unique<
        TimeIntegration::TimeMarchingImplicit<dim,
                                              VectorType,
                                              NonLinearOperator,
                                              ConditionalOStream>>(
        nonlinear_operator,
        *preconditioner,
        *residual_wrapper,
        dof_handler,
        constraints,
        params.nonlinear_data,
        statistics,
        timer,
        pcout,
        std::move(nl_setup_custom_preconditioner),
        std::move(nl_setup_linearization_point),
        std::move(nl_quantities_to_check),
        params.print_time_loop);

      // set initial condition

      std::function<void(VectorType &)> f_init =
        [&nonlinear_operator](VectorType &v) {
          nonlinear_operator.initialize_dof_vector(v);
        };

      solution_history.apply(f_init);
      std::for_each(additional_initializations.begin(),
                    additional_initializations.end(),
                    [](auto &a_init) { a_init(); });

      VectorType &solution = solution_history.get_current_solution();

      bool system_has_changed = true;

      const auto execute_coarsening_and_refinement =
        [&](const double t,
            const double top_fraction_of_cells,
            const double bottom_fraction_of_cells) {
          ScopedName sc("execute_coarsening_and_refinement");
          MyScope    scope(timer, sc);

          pcout << "Execute refinement/coarsening:" << std::endl;

          system_has_changed = true;

          auto solutions_except_recent = solution_history.filter(true, false);
          auto all_blocks = solutions_except_recent.get_all_blocks();
          VectorType vector_solutions_except_recent(all_blocks.begin(),
                                                    all_blocks.end());

          output_result(solution,
                        nonlinear_operator,
                        grain_tracker,
                        advection_mechanism,
                        statistics,
                        t,
                        timer,
                        "refinement");

          // and limit the number of levels
          const unsigned int max_allowed_level =
            (this->n_global_levels_0 - 1) + this->current_max_refinement_depth;
          const unsigned int min_allowed_level =
            (this->n_global_levels_0 - 1) -
            std::min((this->n_global_levels_0 - 1),
                     params.adaptivity_data.min_refinement_depth);

          std::function<void(VectorType &)> after_amr = [&](VectorType &v) {
            initialize(solution.n_blocks());

            nonlinear_operator.clear();
            time_marching->clear();
            preconditioner->clear();

            mass_operator.clear();
            if (params.output_data.fluxes_divergences)
              postproc_preconditioner->clear();

            for (unsigned int i = 0; i < v.n_blocks();
                 i += nonlinear_operator.n_components())
              {
                auto subvector =
                  v.create_view(i, i + nonlinear_operator.n_components());
                nonlinear_operator.initialize_dof_vector(*subvector);
              }
          };

          const unsigned int block_estimate_start = order_parameters_offset;
          const unsigned int block_estimate_end = sintering_data.n_components();
          coarsen_and_refine_mesh(vector_solutions_except_recent,
                                  tria,
                                  dof_handler,
                                  constraints,
                                  Quadrature<dim - 1>(quad),
                                  top_fraction_of_cells,
                                  bottom_fraction_of_cells,
                                  min_allowed_level,
                                  max_allowed_level,
                                  after_amr,
                                  params.adaptivity_data.interface_val_min,
                                  params.adaptivity_data.interface_val_max,
                                  block_estimate_start,
                                  block_estimate_end);

          std::for_each(additional_initializations.begin(),
                        additional_initializations.end(),
                        [](auto &a_init) { a_init(); });

          const auto old_old_solutions = solution_history.filter(false, false);
          old_old_solutions.update_ghost_values();

          if (params.advection_data.enable &&
              params.advection_data.check_courant)
            advection_operator.precompute_cell_diameters();

          output_result(solution,
                        nonlinear_operator,
                        grain_tracker,
                        advection_mechanism,
                        statistics,
                        t,
                        timer,
                        "refinement");

          if (params.output_data.mesh_overhead_estimate)
            Postprocessors::estimate_overhead(mapping, dof_handler, solution);
        };

      // Check consistency of the grain tracker settings
      const bool grain_tracker_required_for_ebc =
        params.boundary_conditions.type == "CentralParticle" ||
        params.boundary_conditions.type == "CentralParticleSection";

      const bool grain_tracker_required_for_output =
        params.output_data.contours || params.output_data.contours_tex ||
        params.output_data.grains_stats ||
        params.output_data.coordination_number;

      const bool grain_tracker_enabled =
        params.grain_tracker_data.grain_tracker_frequency > 0 ||
        (params.adaptivity_data.quality_control &&
         params.grain_tracker_data.track_with_quality) ||
        params.advection_data.enable || grain_tracker_required_for_ebc ||
        grain_tracker_required_for_output ||
        params.grain_tracker_data.check_inconsistency;

      const auto run_grain_tracker = [&](const double t,
                                         const bool   do_initialize = false) {
        ScopedName sc("run_grain_tracker");
        MyScope    scope(timer, sc);

        pcout << "Execute grain tracker("
              << (do_initialize ? "initial_setup" : "track")
              << "):" << std::endl;

        system_has_changed = true;

        auto solutions_except_recent = solution_history.filter(true, false);
        auto old_old_solutions       = solution_history.filter(false, false);

        solutions_except_recent.update_ghost_values();

        output_result(solution,
                      nonlinear_operator,
                      grain_tracker,
                      advection_mechanism,
                      statistics,
                      t,
                      timer,
                      "track");

        const auto time_total = std::chrono::system_clock::now();

        /* The grain tracker design is a bit error prone. If we had
         * initial_setup() or track() calls previously which resulted in some
         * grains reassignments and remap() was not called after that, then the
         * other calls to track() will lead to an error. Ideally, the grains
         * indices should be changed only during the remap() call commiting the
         * new state of the grain tracker.
         *
         * TODO: improve the grain tracker logic
         */
        std::tuple<unsigned int, bool, bool> gt_status;
        if (do_initialize)
          {
            ScopedName sc("setup");
            MyScope    scope(timer, sc);
            gt_status =
              grain_tracker.initial_setup(solution, sintering_data.n_grains());
          }
        else
          {
            ScopedName sc("track");
            MyScope    scope(timer, sc);
            gt_status =
              grain_tracker.track(solution, sintering_data.n_grains());
          }

        const unsigned int n_collisions          = std::get<0>(gt_status);
        const bool         has_reassigned_grains = std::get<1>(gt_status);
        const bool         has_op_number_changed = std::get<2>(gt_status);

        pcout << "\033[96mCollisions detected: " << n_collisions << " | "
              << std::boolalpha
              << "has_reassigned_grains = " << has_reassigned_grains << " | "
              << "has_op_number_changed = " << has_op_number_changed
              << std::noboolalpha << "\033[0m" << std::endl;

        const double time_total_double =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - time_total)
            .count() /
          1e9;

        pcout << "Grain tracker CPU time = "
              << Utilities::MPI::max(time_total_double, MPI_COMM_WORLD)
              << " sec" << std::endl;

        // By default the output is limited
        if (params.grain_tracker_data.verbosity >= 2)
          {
            grain_tracker.print_current_grains(pcout);
          }
        else
          {
            pcout << "Number of order parameters: "
                  << grain_tracker.get_active_order_parameters().size()
                  << std::endl;
            pcout << "Number of grains: " << grain_tracker.get_grains().size()
                  << std::endl;
          }

        // Rebuild data structures if grains have been reassigned
        if (has_reassigned_grains || has_op_number_changed)
          {
            unsigned int n_grains_remapped = 0;

            if (has_op_number_changed)
              {
                const unsigned int n_grains_new =
                  grain_tracker.get_active_order_parameters().size();
                const unsigned int n_components_old = solution.n_blocks();
                const unsigned int n_components_new =
                  n_components_old - sintering_data.n_grains() + n_grains_new;

                pcout << "\033[34mChanging number of components from "
                      << n_components_old << " to " << n_components_new
                      << "\033[0m" << std::endl;

                AssertThrow(n_grains_new <= MAX_SINTERING_GRAINS,
                            ExcMessage("Number of grains (" +
                                       std::to_string(n_grains_new) +
                                       ") exceeds the maximum value (" +
                                       std::to_string(MAX_SINTERING_GRAINS) +
                                       ")."));

                nonlinear_operator.clear();
                time_marching->clear();
                preconditioner->clear();

                /**
                 * If the number of components has reduced, then we remap first
                 * and then alter the number of blocks in the solution vector.
                 * If the number of components has increased, then we need to
                 * add new blocks to the solution vector prior to remapping.
                 */

                const unsigned int distance =
                  nonlinear_operator.n_components() -
                  sintering_data.n_components();

                // Perform remapping
                auto all_solution_vectors =
                  solutions_except_recent.get_all_solutions();

                if (has_reassigned_grains &&
                    n_components_new < n_components_old)
                  {
                    ScopedName sc("remap");
                    MyScope    scope(timer, sc);

                    n_grains_remapped =
                      grain_tracker.remap(all_solution_vectors);

                    // Move the to be deleted components to the end
                    if (distance > 0)
                      for (auto &sol : all_solution_vectors)
                        for (unsigned int i = sintering_data.n_components() - 1;
                             i >= n_components_new;
                             --i)
                          sol->move_block(i, i + distance);
                  }

                // Resize solution vector (delete or add blocks)
                solutions_except_recent.apply(
                  [&](auto &sol) { sol.reinit(n_components_new); });

                if (has_reassigned_grains &&
                    n_components_new > n_components_old)
                  {
                    ScopedName sc("remap");
                    MyScope    scope(timer, sc);

                    // Move the newly created before displacements
                    if (distance > 0)
                      for (auto &sol : all_solution_vectors)
                        for (unsigned int i = nonlinear_operator.n_components();
                             i < n_components_new;
                             ++i)
                          sol->move_block(i, i - distance);

                    n_grains_remapped =
                      grain_tracker.remap(all_solution_vectors);
                  }

                // Change number of components after remapping completed
                sintering_data.set_n_components(n_grains_new +
                                                order_parameters_offset);
              }
            else if (has_reassigned_grains)
              {
                ScopedName sc("remap");
                MyScope    scope(timer, sc);

                // Perform remapping
                auto all_solution_vectors =
                  solutions_except_recent.get_all_solutions();

                n_grains_remapped = grain_tracker.remap(all_solution_vectors);
              }

            pcout << "\033[96mGrains remapped: " << n_grains_remapped << "/"
                  << grain_tracker.get_grains().size() << "\033[0m"
                  << std::endl;

            // We need to call track again if advection mechanism is used
            // in order to keep op_particle_ids in sync
            if (params.advection_data.enable || grain_tracker_required_for_ebc)
              {
                ScopedName sc("track_fast");
                MyScope    scope(timer, sc);

                const bool skip_reassignment = true;
                grain_tracker.track(solution,
                                    sintering_data.n_grains(),
                                    skip_reassignment);
              }

            output_result(solution,
                          nonlinear_operator,
                          grain_tracker,
                          advection_mechanism,
                          statistics,
                          t,
                          timer,
                          "remap");
          }

        pcout << std::endl;

        solution.zero_out_ghost_values();
        old_old_solutions.update_ghost_values();
      };

      // Impose boundary conditions
      const auto impose_boundary_conditions = [&](const double t) {
        (void)t;

        // Update mechanical constraints for the coupled model
        // Currently only fixing the central section along x-axis
        if constexpr (std::is_base_of_v<
                        SinteringOperatorCoupledBase<dim,
                                                     Number,
                                                     VectorizedArrayType,
                                                     NonLinearOperator>,
                        NonLinearOperator>)
          {
            ScopedName sc("impose_boundary_conditions");
            MyScope    scope(timer, sc);

            std::array<bool, dim> directions_mask;
            for (const auto &d : params.boundary_conditions.direction)
              if (d < dim)
                directions_mask[d] = true;

            pcout << "Impose boundary conditions:" << std::endl;
            pcout << "  - type: " << params.boundary_conditions.type
                  << std::endl;
            pcout
              << "  - direction: "
              << debug::to_string(params.boundary_conditions.direction.begin(),
                                  params.boundary_conditions.direction.end())
              << std::endl;

            auto &displ_constraints_indices =
              nonlinear_operator.get_zero_constraints_indices();

            if (params.boundary_conditions.type == "CentralSection")
              {
                clamp_central_section<dim>(displ_constraints_indices,
                                           matrix_free,
                                           mapping,
                                           directions_mask);
              }
            else if (params.boundary_conditions.type == "CentralParticle")
              {
                AssertThrow(
                  grain_tracker_enabled,
                  ExcMessage(
                    "Grain tracker has to be enabled to use CentralParticle"));

                Point<dim> origin = find_center_origin(
                  matrix_free.get_dof_handler().get_triangulation(),
                  grain_tracker,
                  params.boundary_conditions.prefer_growing,
                  params.boundary_conditions.use_barycenter);
                pcout << "  - origin for clamping the section: " << origin
                      << std::endl;

                clamp_section_within_particle<dim>(displ_constraints_indices,
                                                   matrix_free,
                                                   mapping,
                                                   sintering_data,
                                                   grain_tracker,
                                                   solution,
                                                   origin,
                                                   directions_mask);
              }
            else if (params.boundary_conditions.type ==
                     "CentralParticleSection")
              {
                AssertThrow(
                  grain_tracker_enabled,
                  ExcMessage(
                    "Grain tracker has to be enabled to use CentralParticleSection"));

                Point<dim> origin = find_center_origin(
                  matrix_free.get_dof_handler().get_triangulation(),
                  grain_tracker);
                pcout << "  - origin for clamping the sections: " << origin
                      << std::endl;

                clamp_section<dim>(displ_constraints_indices,
                                   matrix_free,
                                   mapping,
                                   origin,
                                   directions_mask);
              }
            else if (params.boundary_conditions.type == "Domain")
              {
                clamp_domain<dim>(displ_constraints_indices, matrix_free);
              }

            pcout << std::endl;
          }
      };

      // Initialize all solutions except the evry old one, it will get
      // overwritten anyway
      initialize_solution(
        solution_history.filter(true, false, true).get_all_blocks_raw(), timer);

      // initial local refinement
      if (t == params.time_integration_data.time_start &&
          (params.adaptivity_data.refinement_frequency > 0 ||
           params.adaptivity_data.quality_control ||
           params.geometry_data.global_refinement != "Full"))
        {
          // Initialize only the current solution
          const auto solution_ptr =
            solution_history.filter(true, false, false).get_all_blocks_raw();

          // If global_refinement is not Full, then we need use more agressive
          // refinement strategy here
          double top_fraction_of_cells =
            params.adaptivity_data.top_fraction_of_cells;
          const double bottom_fraction_of_cells =
            params.adaptivity_data.bottom_fraction_of_cells;
          if (params.geometry_data.global_refinement == "None")
            top_fraction_of_cells *= 3;
          else if (params.geometry_data.global_refinement == "Base")
            top_fraction_of_cells *= 2.5;

          top_fraction_of_cells = std::min(top_fraction_of_cells, 1.0);

          const unsigned int n_init_refinements =
            std::max(std::min(tria.n_global_levels() - 1,
                              params.adaptivity_data.min_refinement_depth),
                     this->n_global_levels_0 - tria.n_global_levels() +
                       params.adaptivity_data.max_refinement_depth);

          pcout << "Number of local refinements to be performed: "
                << n_init_refinements << std::endl;

          for (unsigned int i = 0; i < n_init_refinements; ++i)
            {
              execute_coarsening_and_refinement(t,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells);
              initialize_solution(solution_ptr, timer);
            }

          if (n_init_refinements == 0)
            advection_operator.precompute_cell_diameters();
        }
      else if (params.advection_data.enable &&
               params.advection_data.check_courant)
        {
          // Precompute this here since it is only called inside
          // execute_coarsening_and_refinement() which has been missed
          advection_operator.precompute_cell_diameters();
        }

      // Grain tracker - first run after we have initial configuration defined
      if (grain_tracker_enabled)
        {
          initialize_grain_tracker(grain_tracker,
                                   [&run_grain_tracker,
                                    current_time = t](bool do_initial_setup) {
                                     run_grain_tracker(current_time,
                                                       do_initial_setup);
                                   });
        }

      // Impose boundary conditions
      impose_boundary_conditions(t);

      // Build additional output
      std::function<void(DataOut<dim> & data_out)> additional_output;

      if (params.output_data.fluxes_divergences)
        additional_output =
          [&postproc_operator, &postproc_lhs, this](DataOut<dim> &data_out) {
            postproc_operator.add_data_vectors(data_out, postproc_lhs, {});
          };

      // Initial configuration output
      if (t == params.time_integration_data.time_start &&
          params.output_data.output_time_interval > 0.0)
        {
          if (params.advection_data.enable)
            advection_operator.evaluate_forces(solution);

          output_result(solution,
                        nonlinear_operator,
                        grain_tracker,
                        advection_mechanism,
                        statistics,
                        time_last_output,
                        timer,
                        "solution",
                        additional_output);
        }

      bool has_converged      = true;
      bool do_mesh_refinement = false;
      bool do_grain_tracker   = false;

      const double tol_termination = 1e-9;

      // run time loop
      {
        while (std::abs(t - params.time_integration_data.time_end) >
               tol_termination)
          {
            ScopedName         sc("time_loop");
            TimerOutput::Scope scope(timer(), sc);

            if (has_converged)
              {
                // Perform sanity check
                if (params.time_integration_data.sanity_check_solution &&
                    has_converged)
                  hpsint::limit_vector_values(
                    solution,
                    sintering_data.build_pf_component_mask(
                      solution.n_blocks()));

                if (do_mesh_refinement)
                  {
                    execute_coarsening_and_refinement(
                      t,
                      params.adaptivity_data.top_fraction_of_cells,
                      params.adaptivity_data.bottom_fraction_of_cells);

                    n_timestep_last_amr = n_timestep;
                  }

                if (do_grain_tracker)
                  {
                    try
                      {
                        run_grain_tracker(t, /*do_initialize = */ false);
                        n_timestep_last_gt = n_timestep;
                      }
                    catch (const GrainTracker::ExcGrainsInconsistency &ex)
                      {
                        output_result(solution,
                                      nonlinear_operator,
                                      grain_tracker,
                                      advection_mechanism,
                                      statistics,
                                      time_last_output,
                                      timer,
                                      "grains_inconsistency");

                        pcout << "\033[31m"
                              << "The errors appeared while matching the grains"
                              << "\033[0m" << std::endl;
                        pcout << "Grains from the previous successful GT run:"
                              << std::endl;
                        grain_tracker.print_old_grains(pcout);

                        pcout << "Grains which were successfully assigned:"
                              << std::endl;
                        grain_tracker.print_current_grains(pcout);

                        AssertThrow(false, ExcMessage(ex.what()));
                      }
                  }

                // Impose boundary conditions
                if (do_mesh_refinement)
                  impose_boundary_conditions(t);
              }

            // Update material properties
            sintering_data.set_time(t);

            // Try to extrapolate initial guess
            if (params.time_integration_data.predictor != "None" && t > 0)
              {
                VectorType extrap;
                nonlinear_operator.initialize_dof_vector(extrap);

                if (params.time_integration_data.predictor == "Euler")
                  {
                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      extrap, solution);
                    extrap.sadd(-dt, solution);
                  }
                else if (params.time_integration_data.predictor == "Midpoint")
                  {
                    VectorType midpoint;
                    nonlinear_operator.initialize_dof_vector(midpoint);

                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      midpoint, solution);
                    midpoint.sadd(-dt / 2., solution);

                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      extrap, midpoint);
                    extrap.sadd(-dt, solution);
                  }
                else if (params.time_integration_data.predictor == "Linear" &&
                         sintering_data.time_data.effective_order() > 1)
                  {
                    const auto dt0 = sintering_data.time_data.get_all_dt()[0];
                    const auto dt1 = sintering_data.time_data.get_all_dt()[1];
                    const auto fac = dt0 / dt1;

                    solution_history.extrapolate(extrap, fac);
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }

                solution_history.set_recent_old_solution(solution);

                // Sanity check of the predicted value
                if (params.time_integration_data.sanity_check_predictor)
                  hpsint::limit_vector_values(
                    extrap,
                    sintering_data.build_pf_component_mask(
                      solution.n_blocks()));

                solution = extrap;
              }
            else
              {
                solution_history.set_recent_old_solution(solution);
              }

            if (has_converged)
              {
                if (params.profiling_data.run_vmults && system_has_changed)
                  {
                    ScopedName sc("profiling_vmult");
                    MyScope    scope(timer, sc);

                    const bool old_timing_state =
                      nonlinear_operator.set_timing(false);

                    sintering_data.fill_quadrature_point_values(
                      matrix_free,
                      solution,
                      params.advection_data.enable,
                      save_all_blocks);

                    VectorType dst, src;

                    nonlinear_operator.initialize_dof_vector(dst);
                    nonlinear_operator.initialize_dof_vector(src);

                    const unsigned int n_repetitions = 100;

                    TimerOutput timer(pcout_statistics,
                                      TimerOutput::never,
                                      TimerOutput::wall_times);

                    if (true)
                      {
                        TimerOutput::Scope scope(timer, "vmult_matrixfree");

                        for (unsigned int i = 0; i < n_repetitions; ++i)
                          nonlinear_operator.vmult(dst, src);
                      }

                    if (true)
                      {
                        TimerOutput::Scope scope(timer, "vmult_helmholtz");

                        HelmholtzOperator<dim, Number, VectorizedArrayType>
                          helmholtz_operator(matrix_free, constraints, 1);

                        for (unsigned int i = 0; i < n_repetitions; ++i)
                          for (unsigned int b = 0; b < src.n_blocks(); ++b)
                            helmholtz_operator.vmult(dst.block(b),
                                                     src.block(b));
                      }

                    if (true)
                      {
                        TimerOutput::Scope scope(timer,
                                                 "vmult_vector_helmholtz");

                        HelmholtzOperator<dim, Number, VectorizedArrayType>
                          helmholtz_operator(matrix_free,
                                             constraints,
                                             src.n_blocks());

                        for (unsigned int i = 0; i < n_repetitions; ++i)
                          helmholtz_operator.vmult(dst, src);
                      }

                    if (true)
                      {
                        {
                          TimerOutput::Scope scope(
                            timer, "vmult_matrixbased_assembly");

                          nonlinear_operator.get_system_matrix();
                        }

                        const auto &matrix =
                          nonlinear_operator.get_system_matrix();

                        typename VectorType::BlockType src_, dst_;

                        const auto partitioner =
                          nonlinear_operator.get_system_partitioner();

                        src_.reinit(partitioner);
                        dst_.reinit(partitioner);

                        VectorTools::merge_components_fast(src, src_);

                        {
                          TimerOutput::Scope scope(timer, "vmult_matrixbased");

                          for (unsigned int i = 0; i < n_repetitions; ++i)
                            matrix.vmult(dst_, src_);
                        }

                        VectorTools::split_up_components_fast(dst_, dst);

                        if (params.matrix_based == false)
                          nonlinear_operator.clear_system_matrix();
                      }

                    timer.print_wall_time_statistics(MPI_COMM_WORLD);

                    nonlinear_operator.set_timing(old_timing_state);
                  }
              }

            const auto process_failure = [&](const std::string &message,
                                             const std::string &label) {
              pcout << "\033[31m" << message << "\033[0m" << std::endl;
              dt *= 0.5;
              pcout << "\033[33mReducing timestep, dt = " << dt << "\033[0m"
                    << std::endl;

              n_failed_tries += 1;
              n_failed_linear_iterations += statistics.n_linear_iterations();
              n_failed_non_linear_iterations +=
                statistics.n_newton_iterations();
              n_failed_residual_evaluations +=
                statistics.n_residual_evaluations();

              solution = solution_history.get_recent_old_solution();

              output_result(solution,
                            nonlinear_operator,
                            grain_tracker,
                            advection_mechanism,
                            statistics,
                            time_last_output,
                            timer,
                            label);

              AssertThrow(
                dt > params.time_integration_data.time_step_min,
                ExcMessage("Minimum timestep size exceeded, solution failed!"));

              nonlinear_operator.clear();
              time_marching->clear();
              preconditioner->clear();

              has_converged = false;

              // We do not need to refine the mesh and also no need to run the
              // grain tracker
              do_mesh_refinement = false;
              do_grain_tracker   = false;
            };

            try
              {
                ScopedName sc("newton");
                MyScope    scope(timer, sc);

                // Reset statistics
                statistics.clear();

                if (params.grain_cut_off_tolerance != 0.0)
                  sintering_data.set_component_mask(
                    matrix_free,
                    solution,
                    params.advection_data.enable,
                    save_all_blocks,
                    params.grain_cut_off_tolerance);

                // Check Courant condition
                if (params.advection_data.enable &&
                    params.advection_data.check_courant)
                  {
                    advection_operator.evaluate_forces(solution);

                    AssertThrow(advection_operator.check_courant(dt),
                                ExcCourantConditionViolated());
                  }

                // note: input/output (solution) needs/has the right
                // constraints applied
                time_marching->make_step(solution);

                // We do not commit yet the state since the GT still can fail
                // the tracking stage due to the poor quality of the solution
                const double       t_new          = t + dt;
                const unsigned int n_timestep_new = n_timestep + 1;

                const bool good_iterations =
                  statistics.n_newton_iterations() <
                    params.time_integration_data.desirable_newton_iterations &&
                  statistics.n_linear_iterations() <
                    params.time_integration_data.desirable_linear_iterations;

                // Reset AMR and GT flags
                do_mesh_refinement = false;
                do_grain_tracker   = false;

                // If this was the last step
                const bool is_last_time_step =
                  (std::abs(t_new - params.time_integration_data.time_end) <
                   tol_termination);

                if (!is_last_time_step)
                  {
                    // Coarsen mesh if we have solved everything nicely, we
                    // can do it only if quality control is disabled
                    if (good_iterations &&
                        params.adaptivity_data.extra_coarsening &&
                        !params.adaptivity_data.quality_control &&
                        this->current_max_refinement_depth ==
                          params.adaptivity_data.max_refinement_depth)
                      {
                        --this->current_max_refinement_depth;
                        pcout << "\033[33mReducing mesh quality"
                              << "\033[0m" << std::endl;
                        do_mesh_refinement = true;
                      }

                    // Check if we need to perform AMR. If quality control is
                    // enabled, then AMR frequency setting is not used
                    if (!do_mesh_refinement)
                      {
                        if (params.adaptivity_data.quality_control)
                          {
                            const auto only_order_parameters =
                              solution.create_view(
                                2, sintering_data.n_components());

                            const auto quality =
                              Postprocessors::estimate_mesh_quality_min(
                                dof_handler, *only_order_parameters);

                            do_mesh_refinement =
                              quality < current_min_mesh_quality;
                          }
                        else
                          {
                            do_mesh_refinement =
                              params.adaptivity_data.refinement_frequency > 0 &&
                              (n_timestep_new - n_timestep_last_amr) %
                                  params.adaptivity_data.refinement_frequency ==
                                0;
                          }
                      }

                    // If advection is enabled or certain boundary conditions
                    // are imposed or we force the solver to attempt to recover
                    // from the inconsistency exception, then execute the grain
                    // tracker.
                    if (params.advection_data.enable ||
                        grain_tracker_required_for_ebc ||
                        params.grain_tracker_data.check_inconsistency)
                      do_grain_tracker = true;
                    // If mesh quality control is enabled and grain tracker is
                    // asked to run at the same time, then execute it
                    // synchronously
                    else if (params.adaptivity_data.quality_control &&
                             params.grain_tracker_data.track_with_quality)
                      do_grain_tracker = do_mesh_refinement;
                    // Otherwise use the default frequency settings
                    else
                      do_grain_tracker =
                        params.grain_tracker_data.grain_tracker_frequency > 0 &&
                        (n_timestep_new - n_timestep_last_gt) %
                            params.grain_tracker_data.grain_tracker_frequency ==
                          0;

                    // If we check for inconsistency,
                    // then we run the grain tracker now
                    if (params.grain_tracker_data.check_inconsistency)
                      {
                        run_grain_tracker(t, /*do_initialize = */ false);
                        n_timestep_last_gt = n_timestep_new;

                        // We rerun the grain tracker once again after AMR if
                        // 1) it is synced with mesh, or
                        // 2) advection is enabled, or
                        // 3) this is required for certain EBCs, or
                        // 4) this is required for certain outputs,
                        // otherwise there is no need to do that again.
                        do_grain_tracker =
                          do_mesh_refinement &&
                          ((params.adaptivity_data.quality_control &&
                            params.grain_tracker_data.track_with_quality) ||
                           params.advection_data.enable ||
                           grain_tracker_required_for_ebc ||
                           grain_tracker_required_for_output);
                      }
                    else
                      {
                        // If we have not run the grain tracker now, then we may
                        // need to run it at the beginning of the next step if:
                        // 1) it is synced with mesh, or
                        // 2) advection is enabled, or
                        // 3) this is required for certain EBCs, or
                        // 4) this is required for certain outputs provided AMR
                        // is also to be executed, or
                        // 5) its regular execution is expected according to the
                        // frequency settings.
                        do_grain_tracker =
                          (do_mesh_refinement &&
                           params.adaptivity_data.quality_control &&
                           params.grain_tracker_data.track_with_quality) ||
                          params.advection_data.enable ||
                          (do_mesh_refinement &&
                           grain_tracker_required_for_output) ||
                          grain_tracker_required_for_ebc ||
                          (params.grain_tracker_data.grain_tracker_frequency >
                             0 &&
                           (n_timestep_new - n_timestep_last_gt) %
                               params.grain_tracker_data
                                 .grain_tracker_frequency ==
                             0);
                      }
                  }

                // If we have reached this point, then we are ready to commit
                // the solution
                has_converged = true;
                t             = t_new;
                n_timestep    = n_timestep_new;

                pcout << std::endl;
                pcout << "t = " << t << ", t_n = " << n_timestep
                      << ", dt = " << dt << ":"
                      << " solved in " << statistics.n_newton_iterations()
                      << " Newton iterations and "
                      << statistics.n_linear_iterations()
                      << " linear iterations" << std::endl;

                n_linear_iterations += statistics.n_linear_iterations();
                n_non_linear_iterations += statistics.n_newton_iterations();
                n_residual_evaluations += statistics.n_residual_evaluations();
                max_reached_dt = std::max(max_reached_dt, dt);

                if (!is_last_time_step)
                  {
                    // If solution has converged within a few iterations, we
                    // can increase the timestep
                    if (good_iterations)
                      {
                        dt *= params.time_integration_data.growth_factor;
                        pcout << "\033[32mIncreasing timestep, dt = " << dt
                              << "\033[0m" << std::endl;

                        if (dt > params.time_integration_data.time_step_max)
                          {
                            dt = params.time_integration_data.time_step_max;
                          }
                      }

                    // Adjust timestep if we exceeded time end
                    if (t + dt > params.time_integration_data.time_end)
                      {
                        dt = params.time_integration_data.time_end - t;
                      }
                  }

                if (params.profiling_data.output_memory_consumption &&
                    system_has_changed)
                  {
                    // general-purpose: tria, dofhander, constraints
                    const auto mc_tria =
                      Utilities::MPI::sum<double>(tria.memory_consumption(),
                                                  MPI_COMM_WORLD);
                    const auto mc_dofhandler = Utilities::MPI::sum<double>(
                      dof_handler.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_affine_constraints =
                      Utilities::MPI::sum<double>(
                        constraints.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_matrix_free = Utilities::MPI::sum<double>(
                      matrix_free.memory_consumption(), MPI_COMM_WORLD);

                    // sintering-related data
                    const auto mc_sintering_data = Utilities::MPI::sum<double>(
                      sintering_data.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_nonlinear_operator =
                      Utilities::MPI::sum<double>(
                        nonlinear_operator.memory_consumption(),
                        MPI_COMM_WORLD);
                    const auto mc_solution_history =
                      Utilities::MPI::sum<double>(
                        solution_history.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_preconditioner = Utilities::MPI::sum<double>(
                      preconditioner->memory_consumption(), MPI_COMM_WORLD);

                    const auto mc_total =
                      mc_tria + mc_dofhandler + mc_affine_constraints +
                      mc_matrix_free + mc_nonlinear_operator +
                      mc_solution_history + mc_preconditioner;

                    pcout << "Memory consumption:     " << mc_total / 1e9
                          << std::endl;
                    pcout << " - triangulation:       " << mc_tria / 1e9
                          << std::endl;
                    pcout << " - dof-handler:         " << mc_dofhandler / 1e9
                          << std::endl;
                    pcout << " - affine constraints:  "
                          << mc_affine_constraints / 1e9 << std::endl;
                    pcout << " - matrix-free:         " << mc_matrix_free / 1e9
                          << std::endl;
                    pcout << " - sintering data:      "
                          << mc_sintering_data / 1e9 << std::endl;
                    pcout << " - non-linear operator: "
                          << mc_nonlinear_operator / 1e9 << std::endl;
                    pcout << " - vectors:             "
                          << mc_solution_history / 1e9 << std::endl;
                    pcout << " - preconditioner:      "
                          << mc_preconditioner / 1e9 << std::endl;
                  }

                // Posptrocessing to calculate divergences of fluxes
                if (params.output_data.fluxes_divergences)
                  {
                    ScopedName sc("fluxes_divergences");
                    MyScope    scope(timer, sc);

                    postproc_preconditioner->do_update();

                    postproc_operator.evaluate_rhs(postproc_rhs, solution);

                    for (unsigned int b = 0; b < postproc_lhs.n_blocks(); ++b)
                      {
                        const auto view_lhs =
                          postproc_lhs.create_view(b, b + 1);
                        const auto view_rhs =
                          postproc_rhs.create_view(b, b + 1);
                        postproc_linear_solver->solve(*view_lhs, *view_rhs);
                      }
                  }

                // Print grain forces
                if (params.advection_data.enable &&
                    params.grain_tracker_data.verbosity >= 2)
                  advection_mechanism.print_forces(pcout, grain_tracker);
              }
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &e)
              {
                process_failure(e.message(), "newton_not_converged");
              }
            catch (const SolverControl::NoConvergence &)
              {
                process_failure("Linear solver did not converge",
                                "linear_solver_not_converged");
              }
            catch (const ExcCourantConditionViolated &e)
              {
                process_failure(e.message(), "courant_violated");
              }
            catch (const GrainTracker::ExcGrainsInconsistency &e)
              {
                process_failure(e.what(), "grains_inconsistency");
              }

            if (has_converged)
              {
                // Commit solutions across time history
                solution_history.commit_old_solutions();

                // If this was the last step
                const bool is_last_time_step =
                  (std::abs(t - params.time_integration_data.time_end) <
                   tol_termination);

                // Output results
                if ((params.output_data.output_time_interval > 0.0) &&
                    (t > params.output_data.output_time_interval +
                           time_last_output ||
                     is_last_time_step))
                  {
                    time_last_output = t;
                    output_result(solution,
                                  nonlinear_operator,
                                  grain_tracker,
                                  advection_mechanism,
                                  statistics,
                                  time_last_output,
                                  timer,
                                  "solution",
                                  additional_output);
                  }

                // Save restart point
                if (is_last_time_step || restart_predicate.now(t))
                  {
                    ScopedName sc("restart");
                    MyScope    scope(timer, sc);

                    unsigned int current_restart_count = restart_counter++;

                    if (params.restart_data.max_output != 0)
                      current_restart_count =
                        current_restart_count % params.restart_data.max_output;

                    const std::string prefix =
                      params.restart_data.prefix + "_" +
                      std::to_string(current_restart_count);

                    pcout << "Saving restart data at t = " << t << " ("
                          << prefix << ")" << std::endl;

                    std::vector<const typename VectorType::BlockType *>
                      solution_ptr;

                    if (params.restart_data.full_history)
                      {
                        auto all_except_old =
                          solution_history.filter(true, false, true);

                        const auto history_all_blocks =
                          all_except_old.get_all_blocks_raw();
                        solution_ptr.assign(history_all_blocks.begin(),
                                            history_all_blocks.end());
                      }
                    else
                      {
                        solution_ptr.resize(solution.n_blocks());
                        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
                          solution_ptr[b] = &solution.block(b);
                      }

                    if (params.restart_data.flexible_output)
                      {
                        if (!solution.has_ghost_elements())
                          solution.update_ghost_values();

                        parallel::distributed::
                          SolutionTransfer<dim, typename VectorType::BlockType>
                            solution_transfer(dof_handler);

                        solution_transfer.prepare_for_serialization(
                          solution_ptr);
                        tria.save(prefix + "_tria");

                        solution.zero_out_ghost_values();
                      }
                    else
                      {
                        parallel::distributed::SolutionSerialization<
                          dim,
                          typename VectorType::BlockType>
                          solution_serialization(dof_handler);

                        solution_serialization.add_vectors(solution_ptr);

                        solution_serialization.save(prefix + "_vectors");
                        tria.save(prefix + "_tria");
                      }

                    std::ofstream out_stream(prefix + "_driver");
                    boost::archive::binary_oarchive fosb(out_stream);
                    fosb << params.restart_data.flexible_output;
                    fosb << params.restart_data.full_history;
                    fosb << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
                    fosb << sintering_data.n_components();
                    fosb << solution.n_blocks();
                    fosb << static_cast<unsigned int>(solution_ptr.size());

                    fosb << *this;

                    // Save grains
                    std::ofstream out_stream_gt(prefix + "_grains");
                    boost::archive::binary_oarchive       fogt(out_stream_gt);
                    std::vector<GrainTracker::Grain<dim>> grains;
                    grain_tracker.save_grains(std::back_inserter(grains));
                    fogt << grains;

                    // Save time data
                    std::ofstream out_stream_time(prefix + "_time");
                    boost::archive::binary_oarchive fos_time(out_stream_time);
                    fos_time << sintering_data.time_data;
                  }

                /* If the mesh quality control is enabled we then perform one of
                 * the 2 actions. Depending on the user preferences, we either
                 * increase refine the mesh if the initial quality was below the
                 * predefined minimum threshold, or we use the initial quality
                 * as the reference value for the simulation. We can do this
                 * check only after the first time has converged since the
                 * diffuse interfaces have to be adjusted in accordance with the
                 * energy properties. */
                if (n_timestep == 1 && params.adaptivity_data.quality_control)
                  {
                    const auto only_order_parameters =
                      solution.create_view(2, sintering_data.n_components());

                    auto quality = Postprocessors::estimate_mesh_quality_min(
                      dof_handler, *only_order_parameters);

                    pcout << std::endl;
                    pcout
                      << "Setting up mesh quality settings after the first step:"
                      << std::endl;
                    pcout << "mesh quality = " << quality << ", quality_min = "
                          << params.adaptivity_data.quality_min << std::endl;

                    if (params.adaptivity_data.auto_quality_min)
                      {
                        current_min_mesh_quality =
                          params.adaptivity_data.auto_quality_initial_factor *
                          quality;
                        pcout
                          << "Using the initial quality to set up quality_min = "
                          << current_min_mesh_quality << std::endl;
                      }
                    else
                      {
                        current_min_mesh_quality =
                          params.adaptivity_data.quality_min;
                        pcout
                          << "Adjusting mesh to meet the quality requirements"
                          << std::endl;
                        while (quality < current_min_mesh_quality)
                          {
                            // If that did not help, we need to allow for finer
                            // cells
                            pcout
                              << "\033[33mIncreasing max_refinement_depth from "
                              << current_max_refinement_depth << " to "
                              << (current_max_refinement_depth + 1) << "\033[0m"
                              << std::endl;

                            ++current_max_refinement_depth;

                            execute_coarsening_and_refinement(
                              t,
                              params.adaptivity_data.top_fraction_of_cells,
                              params.adaptivity_data.bottom_fraction_of_cells);

                            quality = Postprocessors::estimate_mesh_quality_min(
                              dof_handler, *only_order_parameters);

                            pcout
                              << "mesh quality = " << quality
                              << ", quality_min = " << current_min_mesh_quality
                              << std::endl;

                            n_timestep_last_amr = n_timestep;
                          }
                      }
                    pcout << std::endl;
                  }

                // Update and advance time step size
                sintering_data.time_data.update_dt(dt);
              }
            else
              {
                // Replace the current time step size
                sintering_data.time_data.replace_dt(dt);
              }

            TimerCollection::print_all_wall_time_statistics();

            system_has_changed = false;
          }
      }

      // clang-format off
      pcout_statistics << std::endl;
      pcout_statistics << "Final statistics:" << std::endl;
      pcout_statistics << "  - n timesteps:               " << n_timestep << std::endl;
      pcout_statistics << "  - n non-linear iterations:   " << n_non_linear_iterations << std::endl;
      pcout_statistics << "  - n linear iterations:       " << n_linear_iterations << std::endl;
      pcout_statistics << "  - n residual evaluations:    " << n_residual_evaluations << std::endl;
      pcout_statistics << "  - avg non-linear iterations: " << static_cast<double>(n_non_linear_iterations) / n_timestep << std::endl;
      pcout_statistics << "  - avg linear iterations:     " << static_cast<double>(n_linear_iterations) / n_non_linear_iterations << std::endl;
      pcout_statistics << "  - max dt:                    " << max_reached_dt << std::endl;
      pcout_statistics << std::endl;
      pcout_statistics << "  - n failed tries:                   " << n_failed_tries << std::endl;
      pcout_statistics << "  - n failed non-linear iterations:   " << n_failed_non_linear_iterations << std::endl;
      pcout_statistics << "  - n failed linear iterations:       " << n_failed_linear_iterations << std::endl;
      pcout_statistics << "  - n failed residual evaluations:    " << n_failed_residual_evaluations << std::endl;      
      pcout_statistics << std::endl;
      // clang-format on

      TimerCollection::print_all_wall_time_statistics(true);
    }

  private:
    void
    output_result(
      const VectorType                   &solution,
      const NonLinearOperator            &sintering_operator,
      GrainTracker::Tracker<dim, Number> &grain_tracker,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
                                                        &advection_mechanism,
      const NonLinearSolvers::NewtonSolverSolverControl &statistics,
      const double                                       t,
      MyTimerOutput                                     &timer,
      const std::string                                  label = "solution",
      std::function<void(DataOut<dim> &data_out)>        additional_output = {})
    {
      if (!params.output_data.debug && label != "solution")
        return; // nothing to do for debug for non-solution

      ScopedName sc("output_result");
      MyScope    scope(timer, sc);

      if (counters.find(label) == counters.end())
        counters[label] = 0;

      // Create table handler
      TableHandler table;

      // Initialize all sections - relevant for 3D case only
      std::vector<Postprocessors::ProjectedData<dim - 1, Number>> sections;

      const std::unordered_map<std::string, int> directions = {{"x", 0},
                                                               {"y", 1},
                                                               {"z", 2}};

      // A separate if to exclude this part for 2D for the code
      if constexpr (dim == 3)
        if (params.output_data.regular || params.output_data.contours ||
            params.output_data.contours_tex)
          {
            for (auto &[direction_code, location] : params.output_data.sections)
              {
                // Data to define cross-sections
                Point<dim> origin;
                origin[directions.at(direction_code)] = location;

                Point<dim> normal;
                normal[directions.at(direction_code)] = 1;

                sections.emplace_back(Postprocessors::build_projection(
                                        dof_handler,
                                        solution,
                                        directions.at(direction_code),
                                        location,
                                        params.output_data.n_coarsening_steps),
                                      origin,
                                      normal);
              }
          }

      if (params.output_data.regular || label != "solution")
        {
          std::vector<std::string> names(solution.n_blocks());

          if (params.output_data.fields.count("CH") &&
              sintering_operator.get_data().n_non_grains() > 0)
            {
              names[0] = "c";
              names[1] = "mu";
            }

          if (params.output_data.fields.count("AC"))
            for (unsigned int ig = 0;
                 ig < sintering_operator.get_data().n_grains();
                 ++ig)
              names[sintering_operator.get_data().n_non_grains() + ig] =
                "eta" + std::to_string(ig);

          if (params.output_data.fields.count("displ") &&
              sintering_operator.n_components() >
                sintering_operator.get_data().n_components())
            for (unsigned int b = solution.n_blocks() - dim;
                 b < solution.n_blocks();
                 ++b)
              names[b] = "u";

          auto data_out = Postprocessors::build_default_output(
            dof_handler,
            solution,
            names,
            params.output_data.fields.count("subdomain"),
            params.output_data.higher_order_cells);

          // Add data provided by the sintering operator
          sintering_operator.add_data_vectors(data_out,
                                              solution,
                                              params.output_data.fields);

          // Output additional data
          if (additional_output)
            additional_output(data_out);

          // Output matrix_free indices - might be needed for debug purposes
          if (params.output_data.fields.count("mf_indices"))
            MyMatrixFreeTools::add_mf_indices_vector(matrix_free, data_out);

          if (params.output_data.fields.count("trans"))
            Postprocessors::add_translation_velocities_vectors(
              matrix_free,
              advection_mechanism,
              sintering_operator.get_data(),
              data_out);

          data_out.build_patches(mapping, this->fe->tensor_degree());

          const std::string output = params.output_data.vtk_path + "/" + label +
                                     "." + std::to_string(counters[label]) +
                                     ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

          // Output sections for 3D case
          if constexpr (dim == 3)
            for (unsigned int i = 0; i < sections.size(); ++i)
              {
                auto proj_data_out = Postprocessors::build_default_output(
                  sections[i].state->dof_handler,
                  Postprocessors::BlockVectorWrapper<
                    std::vector<Vector<Number>>>(sections[i].state->solution),
                  names,
                  params.output_data.fields.count("subdomain"),
                  params.output_data.higher_order_cells);

                if (sections[i].state->dof_handler.n_dofs() > 0)
                  proj_data_out.build_patches();

                std::stringstream ss;
                ss << params.output_data.vtk_path << "/section_"
                   << params.output_data.sections[i].first << "="
                   << params.output_data.sections[i].second << "_" << label
                   << "." << counters[label] << ".vtu";

                const std::string output = ss.str();

                pcout << "Outputing data at t = " << t << " (" << output << ")"
                      << std::endl;

                proj_data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
              }
        }

      // Bounding boxes can be used for scalar table output and for the surface
      // and gb contours
      std::vector<std::shared_ptr<const BoundingBoxFilter<dim>>> box_filters;

      // If we need to add the automatic box
      if (params.output_data.auto_control_box)
        {
          const auto packing_size = geometry_domain_boundaries.second -
                                    geometry_domain_boundaries.first;
          const auto internal_padding =
            params.output_data.box_rel_padding * packing_size;

          auto auto_box_size = packing_size - internal_padding;
          for (unsigned int d = 0; d < dim; ++d)
            auto_box_size[d] -= 2 * geometry_r_max;

          const auto has_negative_dim =
            std::any_of(auto_box_size.begin_raw(),
                        auto_box_size.end_raw(),
                        [](const auto &val) { return val <= 0; });

          if (!has_negative_dim)
            {
              auto boundaries = geometry_domain_boundaries;

              for (unsigned int d = 0; d < dim; ++d)
                {
                  boundaries.first[d] += internal_padding[d] + geometry_r_max;
                  boundaries.second[d] -= internal_padding[d] + geometry_r_max;
                }

              const BoundingBox<dim> control_box(
                std::make_pair(boundaries.first, boundaries.second));
              box_filters.push_back(
                std::make_shared<const BoundingBoxFilter<dim>>(control_box));
            }
        }

      // Add also each individual bounding box
      for (const auto &pp : params.output_data.control_boxes)
        {
          // We can not directly use pp since it always contains objects of type
          // Point<3> and not Point<dim> as one may think. This is done in order
          // to use the same input file for both 2D and 3D cases.
          Point<dim> bottom_left;
          Point<dim> top_right;

          for (unsigned int d = 0; d < dim; ++d)
            {
              bottom_left[d] = pp.first[d];
              top_right[d]   = pp.second[d];
            }

          const auto box_size = top_right - bottom_left;

          const auto has_negative_dim =
            std::any_of(box_size.begin_raw(),
                        box_size.end_raw(),
                        [](const auto &val) { return val <= 0; });

          if (!has_negative_dim)
            {
              const BoundingBox<dim> control_box(
                std::make_pair(bottom_left, top_right));
              box_filters.push_back(
                std::make_shared<const BoundingBoxFilter<dim>>(control_box));
            }
        }

      // Flag if we have any real control boxes
      const bool has_control_boxes = !box_filters.empty();

      // Add empty dummy box if we output data also for the whole domain at the
      // beginning of the vector. The vector is not large, so we can tolerate
      // this expensive insertion.
      if (!has_control_boxes || !params.output_data.only_control_boxes)
        box_filters.insert(box_filters.begin(), nullptr);

      // We need to update the grain tracker data without remapping for these 3
      // kinds of output. Note, we do that only if advection is disabled, then
      // there is a possbility that grain tracker runs not every timestep. If
      // advection is enabled, then we do not need to execute track(). Moreover,
      // calling track() without a subsequent re-evaluation of the grain forces,
      // may result in inconsistency of the data structures which store forces
      // data. This is a design issue, it needs to be fixed, but so far we just
      // keep it in mind.
      if ((params.output_data.contours || params.output_data.contours_tex ||
           params.output_data.grains_stats ||
           params.output_data.coordination_number) &&
          !grain_tracker.empty() && !advection_mechanism.enabled())
        {
          const bool skip_reassignment = true;
          grain_tracker.track(solution,
                              sintering_operator.n_grains(),
                              skip_reassignment);
        }

      const std::function<std::string(const std::string &, const unsigned int)>
        generate_name = [&](const std::string &name, const unsigned int index) {
          if (!has_control_boxes)
            return name;
          else if (params.output_data.only_control_boxes)
            return name + "_" + std::to_string(index);
          else
            return (index == 0) ? name :
                                  (name + "_" + std::to_string(index - 1));
        };

      if (params.output_data.table)
        {
          table.add_value("step", counters[label]);

          table.add_value("time", t);
          if (t < 1.)
            table.set_scientific("time", true);

          const auto current_dt =
            sintering_operator.get_data().time_data.get_current_dt();
          table.add_value("dt", current_dt);
          if (current_dt < 1.)
            table.set_scientific("dt", true);

          table.add_value("n_dofs", dof_handler.n_dofs());
          table.add_value("n_op",
                          sintering_operator.get_data().n_components() - 2);
          table.add_value("n_grains", grain_tracker.get_grains().size());

          table.add_value("n_newton_iter", statistics.n_newton_iterations());
          table.add_value("n_linear_iter", statistics.n_linear_iterations());
          table.add_value("n_resid_eval", statistics.n_residual_evaluations());

          if (!params.output_data.domain_integrals.empty())
            {
              std::vector<std::string> quantities;
              std::copy(params.output_data.domain_integrals.begin(),
                        params.output_data.domain_integrals.end(),
                        std::back_inserter(quantities));

              auto [q_labels, q_evaluators] =
                sintering_operator.build_domain_quantities_evaluators(
                  quantities);

              // TODO: each quantity should provide its flag
              EvaluationFlags::EvaluationFlags eval_flags =
                EvaluationFlags::values | EvaluationFlags::gradients;

              for (unsigned int i = 0; i < box_filters.size(); ++i)
                {
                  std::vector<Number> q_values;

                  if (box_filters[i])
                    {
                      const auto &box_filter = box_filters[i];

                      std::function<VectorizedArrayType(
                        const Point<dim, VectorizedArrayType> &)>
                        predicate_integrals =
                          [&box_filter](
                            const Point<dim, VectorizedArrayType> &p) {
                            return box_filter->filter(p);
                          };

                      q_values = sintering_operator.calc_domain_quantities(
                        q_evaluators,
                        solution,
                        predicate_integrals,
                        eval_flags);
                    }
                  else
                    {
                      q_values =
                        sintering_operator.calc_domain_quantities(q_evaluators,
                                                                  solution,
                                                                  eval_flags);
                    }


                  for (unsigned int j = 0; j < q_evaluators.size(); ++j)
                    table.add_value(generate_name(q_labels[j], i), q_values[j]);
                }
            }

          for (unsigned int i = 0; i < box_filters.size(); ++i)
            {
              if (box_filters[i])
                {
                  const auto box_volume =
                    box_filters[i]->get_bounding_box().volume();

                  table.add_value(generate_name("control_box", i), box_volume);
                }

              if (params.output_data.coordination_number &&
                  !grain_tracker.empty())
                {
                  const auto avg_coord_num =
                    Postprocessors::compute_average_coordination_number(
                      dof_handler,
                      sintering_operator.n_grains(),
                      grain_tracker,
                      box_filters[i]);

                  table.add_value(generate_name("avg_coord_num", i),
                                  avg_coord_num);
                }
            }
        }

      if (params.output_data.shrinkage)
        {
          const std::string output = params.output_data.vtk_path +
                                     "/shrinkage_" + label + "." +
                                     std::to_string(counters[label]) + ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          const auto bb = Postprocessors::estimate_shrinkage(
            mapping,
            dof_handler,
            solution,
            params.output_data.shrinkage_intervals);

          Postprocessors::write_bounding_box(bb, mapping, dof_handler, output);

          if (params.output_data.table)
            {
              const std::vector labels = {"dim_x", "dim_y", "dim_z"};
              typename VectorType::value_type volume = 1.;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  const auto size = bb.side_length(d);
                  table.add_value(labels[d], size);
                  volume *= size;
                }
              table.add_value("volume", volume);
            }
        }

      // Estimate mesh quality
      const auto only_order_parameters =
        solution.create_view(2, sintering_operator.get_data().n_components());
      if (params.output_data.quality)
        {
          const std::string output = params.output_data.vtk_path +
                                     "/mesh_quality_" + label + "." +
                                     std::to_string(counters[label]) + ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          if (params.output_data.quality_min)
            {
              const auto quality = Postprocessors::output_mesh_quality_and_min(
                mapping, dof_handler, *only_order_parameters, output);

              table.add_value("mesh_quality", quality);
            }
          else
            {
              Postprocessors::output_mesh_quality(mapping,
                                                  dof_handler,
                                                  *only_order_parameters,
                                                  output);
            }
        }
      else if (params.output_data.quality_min)
        {
          const auto quality =
            Postprocessors::estimate_mesh_quality_min(dof_handler,
                                                      *only_order_parameters);

          table.add_value("mesh_quality", quality);
        }

      if (params.output_data.grains_stats && !grain_tracker.empty())
        {
          const std::string output = params.output_data.vtk_path +
                                     "/grains_stats_" + label + "." +
                                     std::to_string(counters[label]) + ".log";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          Postprocessors::output_grains_stats(mapping,
                                              dof_handler,
                                              sintering_operator,
                                              grain_tracker,
                                              advection_mechanism,
                                              solution,
                                              output);
        }

      // Advanced output for contours
      if constexpr (dim >= 2)
        Postprocessors::advanced_output(mapping,
                                        dof_handler,
                                        solution,
                                        sintering_operator.n_grains(),
                                        params.output_data,
                                        grain_tracker,
                                        sections,
                                        box_filters,
                                        table,
                                        t,
                                        timer,
                                        generate_name,
                                        counters[label],
                                        label,
                                        pcout);

      if (params.output_data.table)
        {
          const std::string output =
            params.output_data.vtk_path + "/" + label + ".log";
          Postprocessors::write_table(table, t, MPI_COMM_WORLD, output);
        }

      if (params.output_data.grains_as_spheres)
        {
          const std::string output = params.output_data.vtk_path +
                                     "/grains_as_spheres_" + label + "." +
                                     std::to_string(counters[label]) + ".vtu";
          GrainTracker::output_grains_as_spherical_particles(
            grain_tracker.get_grains(), dof_handler, output);
        }

      if (params.output_data.particle_indices)
        {
          const std::string output = params.output_data.vtk_path +
                                     "/particle_ids_" + label + "." +
                                     std::to_string(counters[label]) + ".vtu";
          grain_tracker.output_current_particle_ids(output);
        }

      counters[label]++;
    }


    std::shared_ptr<const FiniteElement<dim>>
    create_fe(const unsigned int fe_degree, const unsigned int n_subdivisions)
    {
      if (n_subdivisions == 1)
        {
          pcout << "Finite element: FE_Q<" << dim << ">, "
                << "n_subdivisions = " << n_subdivisions << std::endl;

          return std::make_shared<FE_Q<dim>>(fe_degree);
        }

      AssertThrow(fe_degree == 1,
                  ExcMessage(
                    "Either fe-degree or number of subdivisions has to be 1."));

      pcout << "Finite element: FE_Q_iso_Q1<" << dim << ">, "
            << "n_subdivisions = " << n_subdivisions << std::endl;

      return std::make_shared<FE_Q_iso_Q1<dim>>(n_subdivisions);
    }
  };
} // namespace Sintering
