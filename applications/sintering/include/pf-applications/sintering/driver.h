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
#include <pf-applications/sintering/preconditioners.h>
#include <pf-applications/sintering/tools.h>

// #define DEBUG_PARAVIEW
#include <pf-applications/grain_tracker/tracker.h>

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

    AffineConstraints<Number> constraint;

    const std::vector<const AffineConstraints<double> *> constraints;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    std::shared_ptr<InitialValues<dim>> initial_solution;

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
      , constraints{&constraint}
      , initial_solution(initial_solution)
    {
      std::pair<dealii::Point<dim>, dealii::Point<dim>> boundaries;

      if (!params.geometry_data.custom_bounding_box)
        {
          boundaries  = initial_solution->get_domain_boundaries();
          double rmax = initial_solution->get_r_max();

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
                  initial_solution->get_interface_width(),
                  params.geometry_data.elements_per_interface,
                  params.geometry_data.periodic);

      initialize();
    }

    void
    initialize()
    {
      // setup DoFHandlers, ...
      dof_handler.distribute_dofs(fe);

      // ... constraints, and ...
      constraint.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);

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
                                                           constraint);
        }

      constraint.close();

      // ... MatrixFree
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;

      matrix_free.reinit(
        mapping, dof_handler, constraint, quad, additional_data);

      // clang-format off
      pcout_statistics << "System statistics:" << std::endl;
      pcout_statistics << "  - n cell:                    " << tria.n_global_active_cells() << std::endl;
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;
      pcout_statistics << "  - n dofs:                    " << dof_handler.n_dofs() << std::endl;
      pcout_statistics << std::endl;
      // clang-format on
    }

    void
    run()
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

      // ... non-linear operator
      NonLinearOperator nonlinear_operator(matrix_free,
                                           constraints,
                                           sintering_data,
                                           params.matrix_based);

      nonlinear_operator.set_n_components(initial_solution->n_components());

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (params.preconditioners_data.outer_preconditioner ==
          "BlockPreconditioner2")
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.preconditioners_data.block_preconditioner_2_data);
      else if (params.preconditioners_data.outer_preconditioner ==
               "BlockPreconditioner3")
        preconditioner = std::make_unique<
          BlockPreconditioner3<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.preconditioners_data.block_preconditioner_3_data);
      else if (params.preconditioners_data.outer_preconditioner ==
               "BlockPreconditioner3CH")
        preconditioner = std::make_unique<
          BlockPreconditioner3CH<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.preconditioners_data.block_preconditioner_3_ch_data);
      else
        preconditioner = Preconditioners::create(
          nonlinear_operator, params.preconditioners_data.outer_preconditioner);

      // ... linear solver
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (true)
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearOperator,
          Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                        *preconditioner);

      TimerOutput timer(pcout_statistics,
                        TimerOutput::never,
                        TimerOutput::wall_times);

      // ... non-linear Newton solver
      auto non_linear_solver =
        std::make_unique<NonLinearSolvers::NewtonSolver<VectorType>>();

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

              nonlinear_operator.evaluate_newton_step(current_u);
            }

          if (do_update_preconditioner)
            {
              MyScope scope(timer, "time_loop::newton::setup_preconditioner");
              preconditioner->do_update();
            }
        };

      non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::solve_with_jacobian");

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraint.set_zero(const_cast<VectorType &>(src).block(b));

        const unsigned int n_iterations = linear_solver->solve(dst, src);

        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraint.distribute(dst.block(b));

        return n_iterations;
      };


      // set initial condition
      VectorType solution;

      nonlinear_operator.initialize_dof_vector(solution);

      const auto initialize_solution = [&]() {
        for (unsigned int c = 0; c < solution.n_blocks(); ++c)
          {
            initial_solution->set_component(c);

            VectorTools::interpolate(mapping,
                                     dof_handler,
                                     *initial_solution,
                                     solution.block(c));

            constraint.distribute(solution.block(c));
          }
        solution.zero_out_ghost_values();
      };

      const unsigned int init_level = tria.n_global_levels() - 1;

      const auto execute_coarsening_and_refinement = [&](const double t) {
        pcout << "Execute refinement/coarsening:" << std::endl;

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
            constraint.distribute(solution_dealii.block(b));
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
               (init_level + params.adaptivity_data.max_refinement_depth)))
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() &&
                   (static_cast<unsigned int>(cell->level()) ==
                    (init_level -
                     std::min(init_level,
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

        initialize();

        nonlinear_operator.clear();
        preconditioner->clear();

        nonlinear_operator.initialize_dof_vector(solution);

        std::vector<typename VectorType::BlockType *> solution_ptr(
          solution.n_blocks());
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          solution_ptr[b] = &solution.block(b);

        solution_trans.interpolate(solution_ptr);

        // note: apply constraints since the Newton solver expects this
        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
          constraint.distribute(solution.block(b));

        output_result(solution, nonlinear_operator, t, "refinement");
      };

      // New grains can not appear in current sintering simulations
      const bool allow_new_grains = false;

      GrainTracker::Tracker<dim, Number> grain_tracker(
        dof_handler,
        !params.geometry_data.minimize_order_parameters,
        allow_new_grains,
        MAX_SINTERING_GRAINS,
        params.grain_tracker_data.threshold_lower,
        params.grain_tracker_data.threshold_upper,
        params.grain_tracker_data.buffer_distance_ratio,
        2);

      const auto run_grain_tracker = [&](const double t,
                                         const bool   do_initialize = false) {
        pcout << "Execute grain tracker:" << std::endl;

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

                nonlinear_operator.set_n_components(n_components_new);
                nonlinear_operator.clear();
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

      initialize_solution();

      // Grain tracker - first run after we have initial configuration defined
      if (params.grain_tracker_data.grain_tracker_frequency > 0)
        {
          run_grain_tracker(0.0, /*do_initialize = */ true);
        }

      // initial local refinement
      if (params.adaptivity_data.refinement_frequency > 0)
        for (unsigned int i = 0;
             i < std::max(params.adaptivity_data.min_refinement_depth,
                          params.adaptivity_data.max_refinement_depth);
             ++i)
          {
            execute_coarsening_and_refinement(0.0);
          }

      double time_last_output = 0;

      if (params.time_integration_data.output_time_interval > 0.0)
        output_result(solution, nonlinear_operator, time_last_output);

      unsigned int n_timestep              = 0;
      unsigned int n_linear_iterations     = 0;
      unsigned int n_non_linear_iterations = 0;
      double       max_reached_dt          = 0.0;

      // run time loop
      {
        TimerOutput::Scope scope(timer, "time_loop");
        for (double t = 0, dt = params.time_integration_data.time_step_init;
             t <= params.time_integration_data.time_end;)
          {
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

            nonlinear_operator.set_timestep(dt);
            nonlinear_operator.set_previous_solution(solution);

            bool has_converged = false;

            try
              {
                MyScope scope(timer, "time_loop::newton");

                // note: input/output (solution) needs/has the right
                // constraints applied
                const auto statistics = non_linear_solver->solve(solution);

                has_converged = true;

                pcout << "t = " << t << ", t_n = " << n_timestep
                      << ", dt = " << dt << ":"
                      << " solved in " << statistics.newton_iterations
                      << " Newton iterations and "
                      << statistics.linear_iterations << " linear iterations"
                      << std::endl;

                n_timestep += 1;
                n_linear_iterations += statistics.linear_iterations;
                n_non_linear_iterations += statistics.newton_iterations;
                max_reached_dt = std::max(max_reached_dt, dt);

                if (std::abs(t - params.time_integration_data.time_end) > 1e-9)
                  {
                    if (statistics.newton_iterations <
                          params.time_integration_data
                            .desirable_newton_iterations &&
                        statistics.linear_iterations <
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
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &)
              {
                dt *= 0.5;
                pcout
                  << "\033[31mNon-linear solver did not converge, reducing timestep, dt = "
                  << dt << "\033[0m" << std::endl;

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

            if ((params.time_integration_data.output_time_interval > 0.0) &&
                has_converged &&
                (t > params.time_integration_data.output_time_interval +
                       time_last_output))
              {
                time_last_output = t;
                output_result(solution, nonlinear_operator, time_last_output);
              }
          }
      }

      // clang-format off
      pcout_statistics << std::endl;
      pcout_statistics << "Final statistics:" << std::endl;
      pcout_statistics << "  - n timesteps:               " << n_timestep << std::endl;
      pcout_statistics << "  - n non-linear iterations:   " << n_non_linear_iterations << std::endl;
      pcout_statistics << "  - n linear iterations:       " << n_linear_iterations << std::endl;
      pcout_statistics << "  - avg non-linear iterations: " << static_cast<double>(n_non_linear_iterations) / n_timestep << std::endl;
      pcout_statistics << "  - avg linear iterations:     " << static_cast<double>(n_linear_iterations) / n_non_linear_iterations << std::endl;
      pcout_statistics << "  - max dt:                    " << max_reached_dt << std::endl;
      pcout_statistics << std::endl;
      // clang-format on

      timer.print_wall_time_statistics(MPI_COMM_WORLD);

      {
        nonlinear_operator.set_timestep(
          params.time_integration_data.time_step_init);
        nonlinear_operator.set_previous_solution(solution);
        nonlinear_operator.evaluate_newton_step(solution);

        VectorType dst, src;

        nonlinear_operator.initialize_dof_vector(dst);
        nonlinear_operator.initialize_dof_vector(src);

        const unsigned int n_repetitions = 1000;

        TimerOutput timer(pcout_statistics,
                          TimerOutput::never,
                          TimerOutput::wall_times);

        {
          TimerOutput::Scope scope(timer, "vmult_matrixfree");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            nonlinear_operator.vmult(dst, src);
        }

        {
          AssertDimension(dst.n_blocks(), 1);
          AssertDimension(src.n_blocks(), 1);

          const auto &matrix = nonlinear_operator.get_system_matrix();

          TimerOutput::Scope scope(timer, "vmult_matrixbased");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            matrix.vmult(dst.block(0), src.block(0));
        }

        timer.print_wall_time_statistics(MPI_COMM_WORLD);
      }
    }

  private:
    void
    output_result(const VectorType &       solution,
                  const NonLinearOperator &sintering_operator,
                  const double             t,
                  const std::string        label = "solution")
    {
#ifndef DEBUG_PARAVIEW
      if (label != "solution")
        return;
#endif

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, solution.block(0), "c");
      data_out.add_data_vector(dof_handler, solution.block(1), "mu");

      for (unsigned int ig = 2; ig < solution.n_blocks(); ++ig)
        data_out.add_data_vector(dof_handler,
                                 solution.block(ig),
                                 "eta" + std::to_string(ig - 2));

      sintering_operator.add_data_vectors(data_out, solution);

      // Output subdomain structure
      Vector<float> subdomain(dof_handler.get_triangulation().n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain[i] =
            dof_handler.get_triangulation().locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");

      data_out.build_patches(mapping, this->fe.tensor_degree());

      static std::map<std::string, unsigned int> counters;

      if (counters.find(label) == counters.end())
        counters[label] = 0;

      std::string output =
        label + "." + std::to_string(counters[label]++) + ".vtu";

      pcout << "Outputing at t = " << t << " (" << output << ")" << std::endl;

      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };
  };
} // namespace Sintering
