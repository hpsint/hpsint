// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/base/fe_integrator.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/vector_tools.h>

#include <pf-applications/sintering/initial_values.h>
#include <pf-applications/sintering/operator.h>
#include <pf-applications/sintering/parameters.h>
#include <pf-applications/sintering/postprocessors.h>
#include <pf-applications/sintering/preconditioners.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/constraint_helper.h>

namespace Sintering
{
  using namespace dealii;


  template <int dim,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class Problem
  {
  public:
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using NonLinearOperator =
      SinteringOperator<dim, Number, VectorizedArrayType>;

    const Parameters                          params;
    ConditionalOStream                        pcout;
    ConditionalOStream                        pcout_statistics;
    parallel::distributed::Triangulation<dim> tria;
    FE_Q<dim>                                 fe;
    MappingQ<dim>                             mapping;
    QGauss<dim>                               quad;
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
    double       time_last_output;
    unsigned int n_timestep;
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
    double       dt;

    std::map<std::string, unsigned int> counters;

    Problem(const Parameters &                  params,
            std::shared_ptr<InitialValues<dim>> initial_solution)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , fe(params.approximation_data.fe_degree)
      , mapping(1)
      , quad(params.approximation_data.n_points_1D)
      , dof_handler(tria)
    {
      geometry_domain_boundaries = initial_solution->get_domain_boundaries();
      geometry_r_max             = initial_solution->get_r_max();
      geometry_interface_width   = initial_solution->get_interface_width();

      this->n_global_levels_0              = tria.n_global_levels();
      this->time_last_output               = 0;
      this->n_timestep                     = 0;
      this->n_linear_iterations            = 0;
      this->n_non_linear_iterations        = 0;
      this->n_residual_evaluations         = 0;
      this->n_failed_tries                 = 0;
      this->n_failed_linear_iterations     = 0;
      this->n_failed_non_linear_iterations = 0;
      this->n_failed_residual_evaluations  = 0;
      this->max_reached_dt                 = 0.0;
      this->restart_counter                = 0;
      this->t                              = 0;
      this->dt       = params.time_integration_data.time_step_init;
      this->counters = {};

      create_grid(true);

      initialize();

      const auto initialize_solution = [&](VectorType &   solution,
                                           MyTimerOutput &timer) {
        MyScope scope(timer, "initialize_solution");

        for (unsigned int c = 0; c < solution.n_blocks(); ++c)
          {
            initial_solution->set_component(c);

            VectorTools::interpolate(mapping,
                                     dof_handler,
                                     *initial_solution,
                                     solution.block(c));

            constraints.distribute(solution.block(c));
          }
        solution.zero_out_ghost_values();
      };

      run(initial_solution->n_components(), initialize_solution);
    }

    Problem(const Parameters &params, const std::string &restart_path)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , fe(params.approximation_data.fe_degree)
      , mapping(1)
      , quad(params.approximation_data.n_points_1D)
      , dof_handler(tria)
    {
      // 0) load internal state
      unsigned int                    n_components = 0;
      std::ifstream                   in_stream(restart_path + "_driver");
      boost::archive::binary_iarchive fisb(in_stream);
      fisb >> n_components;
      fisb >> *this;

      // 1) create coarse mesh
      create_grid(false);

      // 2) load mesh refinement (incl. vectors)
      tria.load(restart_path + "_tria");

      // 3) initialize data structures
      initialize();

      // 4) helper function to initialize solution vector
      const auto initialize_solution = [&](VectorType &   solution,
                                           MyTimerOutput &timer) {
        MyScope scope(timer, "deserialize_solution");

        parallel::distributed::SolutionTransfer<dim,
                                                typename VectorType::BlockType>
          solution_transfer(dof_handler);

        std::vector<typename VectorType::BlockType *> solution_ptr(
          solution.n_blocks());
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          solution_ptr[b] = &solution.block(b);

        solution_transfer.deserialize(solution_ptr);
      };

      // 5) run time loop
      run(n_components, initialize_solution);
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
      ar &time_last_output;
      ar &n_timestep;
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
      ar &dt;
      ar &counters;
    }

    void
    create_grid(const bool with_initial_refinement)
    {
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
          if (dim >= 1)
            {
              boundaries.first[0] =
                params.geometry_data.bounding_box_data.x_min;
              boundaries.second[0] =
                params.geometry_data.bounding_box_data.x_max;
            }
          if (dim >= 2)
            {
              boundaries.first[1] =
                params.geometry_data.bounding_box_data.y_min;
              boundaries.second[1] =
                params.geometry_data.bounding_box_data.y_max;
            }
          if (dim == 3)
            {
              boundaries.first[2] =
                params.geometry_data.bounding_box_data.z_min;
              boundaries.second[2] =
                params.geometry_data.bounding_box_data.z_max;
            }
        }

      create_mesh(tria,
                  boundaries.first,
                  boundaries.second,
                  geometry_interface_width,
                  params.geometry_data.elements_per_interface,
                  params.geometry_data.periodic,
                  with_initial_refinement);

      helper = std::make_unique<dealii::parallel::Helper<dim>>(tria);

      const auto weight_function = parallel::hanging_nodes_weighting<dim>(
        *helper, params.geometry_data.hanging_node_weight);
      tria.signals.weight.connect(weight_function);
      tria.repartition();
    }

    void
    initialize(const unsigned int n_components = 0)
    {
      // setup DoFHandlers, ...
      dof_handler.distribute_dofs(fe);

      // ... constraints, and ...
      constraints.clear();
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

          DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                           constraints);
        }

      constraints.close();

      // ... MatrixFree
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;

      matrix_free.reinit(
        mapping, dof_handler, constraints, quad, additional_data);

      if ((params.preconditioners_data.outer_preconditioner == "GMG") ||
          (params.preconditioners_data.outer_preconditioner == "BlockGMG") ||
          ((params.preconditioners_data.outer_preconditioner ==
            "BlockPreconditioner2") &&
           ((params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "GMG") ||
            (params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "BlockGMG"))))
        {
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
              dof_handler.distribute_dofs(fe);

              constraints.clear();
              DoFTools::make_hanging_node_constraints(dof_handler, constraints);
              Assert(params.geometry_data.periodic == false,
                     ExcNotImplemented());
              constraints.close();

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
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;

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
    run(const unsigned int n_intial_components,
        const std::function<void(VectorType &, MyTimerOutput &)>
          &initialize_solution)
    {
      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        params.energy_data.A,
        params.energy_data.B,
        params.mobility_data.Mvol,
        params.mobility_data.Mvap,
        params.mobility_data.Msurf,
        params.mobility_data.Mgb,
        params.mobility_data.L,
        params.energy_data.kappa_c,
        params.energy_data.kappa_p);

      sintering_data.set_n_components(n_intial_components);

      MGLevelObject<SinteringOperatorData<dim, VectorizedArrayType>>
        mg_sintering_data(0,
                          n_global_levels_0 +
                            params.adaptivity_data.max_refinement_depth,
                          sintering_data);

      // ... non-linear operator
      NonLinearOperator nonlinear_operator(matrix_free,
                                           constraints,
                                           sintering_data,
                                           params.matrix_based);

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (transfer)
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          sintering_data,
          matrix_free,
          constraints,
          mg_sintering_data,
          mg_matrix_free,
          mg_constraints,
          transfer,
          params.preconditioners_data.block_preconditioner_2_data);
      else if (params.preconditioners_data.outer_preconditioner ==
               "BlockPreconditioner2")
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          sintering_data,
          matrix_free,
          constraints,
          params.preconditioners_data.block_preconditioner_2_data);
      else
        preconditioner = Preconditioners::create(
          nonlinear_operator, params.preconditioners_data.outer_preconditioner);

      // ... linear solver
      ReductionControl solver_control_l(params.nonlinear_data.l_max_iter,
                                        params.nonlinear_data.l_abs_tol,
                                        params.nonlinear_data.l_rel_tol);
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (true)
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearOperator,
          Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                        *preconditioner,
                                                        solver_control_l);

      MyTimerOutput timer;
      TimerCollection::configure(params.profiling_data.output_time_interval);

      // ... non-linear Newton solver
      NonLinearSolvers::NewtonSolverSolverControl statistics(
        params.nonlinear_data.nl_max_iter,
        params.nonlinear_data.nl_abs_tol,
        params.nonlinear_data.nl_rel_tol);

      auto non_linear_solver =
        std::make_unique<NonLinearSolvers::NewtonSolver<VectorType>>(
          statistics,
          NonLinearSolvers::NewtonSolverAdditionalData(
            params.nonlinear_data.newton_do_update,
            params.nonlinear_data.newton_threshold_newton_iter,
            params.nonlinear_data.newton_threshold_linear_iter,
            params.nonlinear_data.newton_reuse_preconditioner));

      non_linear_solver->reinit_vector = [&](auto &vector) {
        MyScope scope(timer, "time_loop::newton::reinit_vector");

        nonlinear_operator.initialize_dof_vector(vector);
      };

      non_linear_solver->residual = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::residual");

        nonlinear_operator.evaluate_nonlinear_residual(dst, src);
      };

      non_linear_solver->setup_jacobian =
        [&](const auto &current_u, const bool do_update_preconditioner) {
          if (true)
            {
              MyScope scope(timer, "time_loop::newton::setup_jacobian");

              sintering_data.fill_quadrature_point_values(matrix_free,
                                                          current_u);

              nonlinear_operator.do_update();
            }

          if (do_update_preconditioner)
            {
              MyScope scope(timer, "time_loop::newton::setup_preconditioner");

              if (transfer) // update multigrid levels
                {
                  const unsigned int min_level = transfers.min_level();
                  const unsigned int max_level = transfers.max_level();
                  const unsigned int n_blocks  = current_u.n_blocks();

                  for (unsigned int l = min_level; l <= max_level; ++l)
                    mg_sintering_data[l].dt = sintering_data.dt;

                  MGLevelObject<VectorType> mg_current_u(min_level, max_level);

                  // acitve level
                  mg_current_u[max_level].reinit(n_blocks);
                  for (unsigned int b = 0; b < n_blocks; ++b)
                    mg_matrix_free[max_level].initialize_dof_vector(
                      mg_current_u[max_level].block(b));
                  mg_current_u[max_level].copy_locally_owned_data_from(
                    current_u);

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
                        mg_matrix_free[l], mg_current_u[l]);
                    }
                }

              preconditioner->do_update();
            }
        };

      non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::solve_with_jacobian");

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.set_zero(const_cast<VectorType &>(src).block(b));

        const unsigned int n_iterations = linear_solver->solve(dst, src);

        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.distribute(dst.block(b));

        return n_iterations;
      };


      // set initial condition
      VectorType solution;

      nonlinear_operator.initialize_dof_vector(solution);

      bool system_has_changed = true;

      const auto execute_coarsening_and_refinement = [&](const double t) {
        MyScope scope(timer, "execute_coarsening_and_refinement");

        pcout << "Execute refinement/coarsening:" << std::endl;

        system_has_changed = true;

        output_result(solution, nonlinear_operator, t, "refinement");

        // 1) copy solution so that it has the right ghosting
        const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler.locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(dof_handler),
          dof_handler.get_communicator());

        VectorType solution_dealii(solution.n_blocks());

        for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
          {
            solution_dealii.block(b).reinit(partitioner);
            solution_dealii.block(b).copy_locally_owned_data_from(
              solution.block(b));
            constraints.distribute(solution_dealii.block(b));
          }

        solution_dealii.update_ghost_values();

        // 2) estimate errors
        Vector<float> estimated_error_per_cell(tria.n_active_cells());

        for (unsigned int b = 2; b < solution_dealii.n_blocks(); ++b)
          {
            Vector<float> estimated_error_per_cell_temp(tria.n_active_cells());

            KellyErrorEstimator<dim>::estimate(
              this->dof_handler,
              QGauss<dim - 1>(this->dof_handler.get_fe().degree + 1),
              std::map<types::boundary_id, const Function<dim> *>(),
              solution_dealii.block(b),
              estimated_error_per_cell_temp,
              {},
              nullptr,
              0,
              Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

            for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
              estimated_error_per_cell[i] += estimated_error_per_cell_temp[i] *
                                             estimated_error_per_cell_temp[i];
          }

        for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
          estimated_error_per_cell[i] = std::sqrt(estimated_error_per_cell[i]);

        // 3) mark automatically cells for coarsening/refinement, ...
        parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction(
            tria,
            estimated_error_per_cell,
            params.adaptivity_data.top_fraction_of_cells,
            params.adaptivity_data.bottom_fraction_of_cells);

        // make sure that cells close to the interfaces are refined, ...
        Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false || cell->refine_flag_set())
              continue;

            for (unsigned int b = 2; b < solution_dealii.n_blocks(); ++b)
              {
                cell->get_dof_values(solution_dealii.block(b), values);

                for (unsigned int i = 0; i < values.size(); ++i)
                  if (0.05 < values[i] && values[i] < 0.95)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();

                      break;
                    }

                if (cell->refine_flag_set())
                  break;
              }
          }

        // and limit the number of levels
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->refine_flag_set() &&
              (static_cast<unsigned int>(cell->level()) ==
               ((this->n_global_levels_0 - 1) +
                params.adaptivity_data.max_refinement_depth)))
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() &&
                   (static_cast<unsigned int>(cell->level()) ==
                    ((this->n_global_levels_0 - 1) -
                     std::min((this->n_global_levels_0 - 1),
                              params.adaptivity_data.min_refinement_depth))))
            cell->clear_coarsen_flag();

        // 4) perform interpolation and initialize data structures
        tria.prepare_coarsening_and_refinement();

        parallel::distributed::SolutionTransfer<dim,
                                                typename VectorType::BlockType>
          solution_trans(dof_handler);

        std::vector<const typename VectorType::BlockType *> solution_dealii_ptr(
          solution_dealii.n_blocks());
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          solution_dealii_ptr[b] = &solution_dealii.block(b);

        solution_trans.prepare_for_coarsening_and_refinement(
          solution_dealii_ptr);

        tria.execute_coarsening_and_refinement();

        initialize(solution_dealii.n_blocks());

        nonlinear_operator.clear();
        non_linear_solver->clear();
        preconditioner->clear();

        nonlinear_operator.initialize_dof_vector(solution);

        std::vector<typename VectorType::BlockType *> solution_ptr(
          solution.n_blocks());
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          solution_ptr[b] = &solution.block(b);

        solution_trans.interpolate(solution_ptr);

        // note: apply constraints since the Newton solver expects this
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          constraints.distribute(solution.block(b));

        output_result(solution, nonlinear_operator, t, "refinement");
      };

      // New grains can not appear in current sintering simulations
      GrainTracker::Tracker<dim, Number> grain_tracker(
        dof_handler,
        !params.geometry_data.minimize_order_parameters,
        /*allow_new_grains*/ false,
        MAX_SINTERING_GRAINS,
        params.grain_tracker_data.threshold_lower,
        params.grain_tracker_data.threshold_upper,
        params.grain_tracker_data.buffer_distance_ratio,
        2);

      const auto run_grain_tracker = [&](const double t,
                                         const bool   do_initialize = false) {
        MyScope scope(timer, "run_grain_tracker");

        pcout << "Execute grain tracker:" << std::endl;

        system_has_changed = true;

        solution.update_ghost_values();

        const auto [has_reassigned_grains, has_op_number_changed] =
          do_initialize ? grain_tracker.initial_setup(solution) :
                          grain_tracker.track(solution);

        grain_tracker.print_current_grains(pcout);

        // Rebuild data structures if grains have been reassigned
        if (has_reassigned_grains || has_op_number_changed)
          {
            output_result(solution, nonlinear_operator, t, "remap");

            if (has_op_number_changed)
              {
                const unsigned int n_components_new =
                  grain_tracker.get_active_order_parameters().size() + 2;
                const unsigned int n_components_old = solution.n_blocks();

                pcout << "\033[34mChanging number of components from "
                      << n_components_old << " to " << n_components_new
                      << "\033[0m" << std::endl;

                sintering_data.set_n_components(n_components_new);

                nonlinear_operator.clear();
                non_linear_solver->clear();
                preconditioner->clear();

                /**
                 * If the number of components has reduced, then we remap first
                 * and then alter the number of blocks in the solution vector.
                 * If the number of components has increased, then we need to
                 * add new blocks to the solution vector prior to remapping.
                 */

                if (has_reassigned_grains &&
                    n_components_new < n_components_old)
                  grain_tracker.remap(solution);

                solution.reinit(n_components_new);

                if (has_reassigned_grains &&
                    n_components_new > n_components_old)
                  grain_tracker.remap(solution);
              }
            else if (has_reassigned_grains)
              {
                grain_tracker.remap(solution);
              }

            output_result(solution, nonlinear_operator, t, "remap");
          }

        solution.zero_out_ghost_values();
      };

      initialize_solution(solution, timer);

      // Grain tracker - first run after we have initial configuration defined
      if (params.grain_tracker_data.grain_tracker_frequency > 0)
        run_grain_tracker(t, /*do_initialize = */ true);

      // initial local refinement
      if (t == 0.0 && params.adaptivity_data.refinement_frequency > 0)
        for (unsigned int i = 0;
             i < std::max(params.adaptivity_data.min_refinement_depth,
                          params.adaptivity_data.max_refinement_depth);
             ++i)
          execute_coarsening_and_refinement(t);

      if (params.output_data.output_time_interval > 0.0)
        output_result(solution, nonlinear_operator, time_last_output);

      // run time loop
      {
        while (t <= params.time_integration_data.time_end)
          {
            TimerOutput::Scope scope(timer(), "time_loop");

            if (n_timestep != 0 &&
                params.adaptivity_data.refinement_frequency > 0 &&
                n_timestep % params.adaptivity_data.refinement_frequency == 0)
              execute_coarsening_and_refinement(t);

            if (n_timestep != 0 &&
                params.grain_tracker_data.grain_tracker_frequency > 0 &&
                n_timestep %
                    params.grain_tracker_data.grain_tracker_frequency ==
                  0)
              {
                try
                  {
                    run_grain_tracker(t, /*do_initialize = */ false);
                  }
                catch (const GrainTracker::ExcCloudsInconsistency &ex)
                  {
                    output_result(solution,
                                  nonlinear_operator,
                                  time_last_output,
                                  "clouds_inconsistency");

                    grain_tracker.dump_last_clouds();

                    AssertThrow(false, ExcMessage(ex.what()));
                  }
              }

            sintering_data.dt = dt;
            nonlinear_operator.set_previous_solution(solution);

            if (params.profiling_data.run_vmults && system_has_changed)
              {
                MyScope scope(timer, "time_loop::profiling_vmult");

                const bool old_timing_state =
                  nonlinear_operator.set_timing(false);

                sintering_data.fill_quadrature_point_values(matrix_free,
                                                            solution);

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
                        helmholtz_operator.vmult(dst.block(b), src.block(b));
                  }

                if (true)
                  {
                    TimerOutput::Scope scope(timer, "vmult_vector_helmholtz");

                    HelmholtzOperator<dim, Number, VectorizedArrayType>
                      helmholtz_operator(matrix_free,
                                         constraints,
                                         src.n_blocks());

                    for (unsigned int i = 0; i < n_repetitions; ++i)
                      helmholtz_operator.vmult(dst, src);
                  }

                if (true)
                  {
                    const auto &matrix = nonlinear_operator.get_system_matrix();

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

                system_has_changed = false;

                nonlinear_operator.set_timing(old_timing_state);
              }

            bool has_converged = false;

            try
              {
                MyScope scope(timer, "time_loop::newton");

                // note: input/output (solution) needs/has the right
                // constraints applied
                non_linear_solver->solve(solution);

                has_converged = true;

                pcout << "t = " << t << ", t_n = " << n_timestep
                      << ", dt = " << dt << ":"
                      << " solved in " << statistics.n_newton_iterations()
                      << " Newton iterations and "
                      << statistics.n_linear_iterations()
                      << " linear iterations" << std::endl;

                n_timestep += 1;
                n_linear_iterations += statistics.n_linear_iterations();
                n_non_linear_iterations += statistics.n_newton_iterations();
                n_residual_evaluations += statistics.n_residual_evaluations();
                max_reached_dt = std::max(max_reached_dt, dt);

                if (std::abs(t - params.time_integration_data.time_end) > 1e-9)
                  {
                    if (statistics.n_newton_iterations() <
                          params.time_integration_data
                            .desirable_newton_iterations &&
                        statistics.n_linear_iterations() <
                          params.time_integration_data
                            .desirable_linear_iterations)
                      {
                        dt *= params.time_integration_data.growth_factor;
                        pcout << "\033[32mIncreasing timestep, dt = " << dt
                              << "\033[0m" << std::endl;

                        if (dt > params.time_integration_data.time_step_max)
                          {
                            dt = params.time_integration_data.time_step_max;
                          }
                      }

                    if (t + dt > params.time_integration_data.time_end)
                      {
                        dt = params.time_integration_data.time_end - t;
                      }
                  }

                t += dt;
              }
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &e)
              {
                dt *= 0.5;
                pcout << "\033[31m" << e.message()
                      << " Reducing timestep, dt = " << dt << "\033[0m"
                      << std::endl;

                n_failed_tries += 1;
                n_failed_linear_iterations += statistics.n_linear_iterations();
                n_failed_non_linear_iterations +=
                  statistics.n_newton_iterations();
                n_failed_residual_evaluations +=
                  statistics.n_residual_evaluations();

                solution = nonlinear_operator.get_previous_solution();

                output_result(solution,
                              nonlinear_operator,
                              time_last_output,
                              "newton_not_converged");

                AssertThrow(
                  dt > params.time_integration_data.time_step_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }
            catch (const SolverControl::NoConvergence &)
              {
                dt *= 0.5;
                pcout
                  << "\033[33mLinear solver did not converge, reducing timestep, dt = "
                  << dt << "\033[0m" << std::endl;

                n_failed_tries += 1;
                n_failed_linear_iterations += statistics.n_linear_iterations();
                n_failed_non_linear_iterations +=
                  statistics.n_newton_iterations();
                n_failed_residual_evaluations +=
                  statistics.n_residual_evaluations();

                solution = nonlinear_operator.get_previous_solution();

                output_result(solution,
                              nonlinear_operator,
                              time_last_output,
                              "linear_solver_not_converged");

                AssertThrow(
                  dt > params.time_integration_data.time_step_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }

            if ((params.output_data.output_time_interval > 0.0) &&
                has_converged &&
                (t >
                 params.output_data.output_time_interval + time_last_output))
              {
                time_last_output = t;
                output_result(solution, nonlinear_operator, time_last_output);
              }

            if ((n_timestep %
                 static_cast<unsigned int>(params.restart_data.interval)) == 0)
              {
                const bool solution_is_ghosted = solution.has_ghost_elements();

                if (solution_is_ghosted == false)
                  solution.update_ghost_values();

                parallel::distributed::
                  SolutionTransfer<dim, typename VectorType::BlockType>
                    solution_transfer(dof_handler);

                std::vector<const typename VectorType::BlockType *>
                  solution_ptr(solution.n_blocks());
                for (unsigned int b = 0; b < solution.n_blocks(); ++b)
                  solution_ptr[b] = &solution.block(b);

                solution_transfer.prepare_for_serialization(solution_ptr);

                const std::string prefix = params.restart_data.prefix + "_" +
                                           std::to_string(restart_counter++);

                tria.save(prefix + "_tria");

                std::ofstream                   out_stream(prefix + "_driver");
                boost::archive::binary_oarchive fosb(out_stream);
                fosb << solution.n_blocks();
                fosb << *this;

                if (solution_is_ghosted == false)
                  solution.zero_out_ghost_values();
              }

            TimerCollection::print_all_wall_time_statistics();
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
    output_result(const VectorType &       solution,
                  const NonLinearOperator &sintering_operator,
                  const double             t,
                  const std::string        label = "solution")
    {
      if (!params.output_data.debug && label != "solution")
        return; // nothing to do for debug for non-solution

      if (counters.find(label) == counters.end())
        counters[label] = 0;

      if (params.output_data.regular || label != "solution")
        {
          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells =
            params.output_data.higher_order_cells;

          DataOut<dim> data_out;
          data_out.set_flags(flags);

          if (params.output_data.fields.count("CH"))
            {
              data_out.add_data_vector(dof_handler, solution.block(0), "c");
              data_out.add_data_vector(dof_handler, solution.block(1), "mu");
            }

          if (params.output_data.fields.count("AC"))
            {
              for (unsigned int ig = 2; ig < solution.n_blocks(); ++ig)
                data_out.add_data_vector(dof_handler,
                                         solution.block(ig),
                                         "eta" + std::to_string(ig - 2));
            }

          sintering_operator.add_data_vectors(data_out,
                                              solution,
                                              params.output_data.fields);

          // Output subdomain structure
          if (params.output_data.fields.count("subdomain"))
            {
              Vector<float> subdomain(
                dof_handler.get_triangulation().n_active_cells());
              for (unsigned int i = 0; i < subdomain.size(); ++i)
                {
                  subdomain[i] =
                    dof_handler.get_triangulation().locally_owned_subdomain();
                }
              data_out.add_data_vector(subdomain, "subdomain");
            }

          data_out.build_patches(mapping, this->fe.tensor_degree());

          std::string output = params.output_data.vtk_path + "/" + label + "." +
                               std::to_string(counters[label]) + ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
        }

      if (params.output_data.contours || label != "solution")
        {
          std::string output = params.output_data.vtk_path + "/contour_" +
                               label + "." + std::to_string(counters[label]) +
                               ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          Postprocessors::output_grain_contours(
            mapping,
            dof_handler,
            solution,
            0.5,
            output,
            params.output_data.n_coarsening_steps);

          counters[label]++;
        }
    };
  };
} // namespace Sintering
