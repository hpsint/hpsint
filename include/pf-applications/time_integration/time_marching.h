// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <pf-applications/lac/evaluation.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_nonlinear.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

namespace TimeIntegration
{
  template <typename VectorType>
  class TimeMarching
  {
  public:
    virtual ~TimeMarching() = default;

    virtual void
    make_step(VectorType &) const = 0;

    virtual void
    clear() const = 0;
  };

  using namespace dealii;
  using namespace hpsint;
  using namespace NonLinearSolvers;

  template <int dim, typename VectorType, typename NonLinearOperator>
  class TimeMarchingImplicit : public TimeMarching<VectorType>
  {
  public:
    using Number = typename NonLinearOperator::value_type;

    TimeMarchingImplicit(
      NonLinearOperator                           &nonlinear_operator,
      Preconditioners::PreconditionerBase<Number> &preconditioner,
      const ResidualWrapper<Number>               &residual_wrapper,
      const DoFHandler<dim>                       &dof_handler,
      const AffineConstraints<Number>             &constraints,
      const NonLinearData                         &nonlinear_params,
      NewtonSolverSolverControl                   &statistics,
      MyTimerOutput                               &timer,
      std::function<void(const VectorType &)>      setup_custom_preconditioner,
      std::function<void(const VectorType &)>      setup_linearization_point,
      std::function<SolverControl::State(const unsigned int,
                                         const double,
                                         const VectorType &,
                                         const VectorType &)>
        check_iteration_status)
    {
      // Create linear solver control
      if (true) // TODO: make parameter
        {
          solver_control_l =
            std::make_unique<ReductionControl>(nonlinear_params.l_max_iter,
                                               nonlinear_params.l_abs_tol,
                                               nonlinear_params.l_rel_tol);
        }
      else
        {
          solver_control_l = std::make_unique<IterationNumberControl>(10);
        }

      // Enable tracking residual evolution
      if (nonlinear_params.verbosity >= 2) // TODO
        solver_control_l->enable_history_data();

      // Create Jacobian operator
      if (nonlinear_params.jacobi_free == false)
        jacobian_operator =
          std::make_unique<JacobianWrapper<Number, NonLinearOperator>>(
            nonlinear_operator);
      else
        jacobian_operator =
          std::make_unique<JacobianFree<Number, ResidualWrapper<Number>>>(
            residual_wrapper);

      auto &j_op = *jacobian_operator;

      // Create linear solver
      linear_solver = LinearSolvers::create(j_op,
                                            preconditioner,
                                            nonlinear_operator,
                                            *solver_control_l,
                                            nonlinear_params,
                                            nonlinear_params.l_solver);


      // Lambda to compute residual
      const auto nl_residual = [&](const auto &src, auto &dst) {
        ScopedName sc("residual");
        MyScope    scope(timer, sc);

        residual_wrapper.evaluate_nonlinear_residual(dst, src);

        statistics.increment_residual_evaluations(1);
      };

      // Lambda to set up jacobian
      const auto nl_setup_jacobian =
        [&, slp = setup_linearization_point](const auto &current_u) {
          ScopedName sc("setup_jacobian");
          MyScope    scope(timer, sc);

          slp(current_u);

          // TODO disable this feature for a while, fix later
          // nonlinear_operator.update_state(current_u);

          nonlinear_operator.do_update();

          if (nonlinear_params.fdm_jacobian_approximation)
            {
              nonlinear_operator.initialize_system_matrix(false);
              auto &system_matrix = nonlinear_operator.get_system_matrix();

              calc_numeric_tangent(dof_handler,
                                   nonlinear_operator,
                                   current_u,
                                   nl_residual,
                                   system_matrix);
            }

          j_op.reinit(current_u);
        };

      // Lambda to update preconditioner
      const auto nl_setup_preconditioner =
        [&, scp = setup_custom_preconditioner](const auto &current_u) {
          ScopedName sc("setup_preconditioner");
          MyScope    scope(timer, sc);

          scp(current_u);

          // Update the underlying system if a preconditioner uses it. The use
          // of the numerical FDM approximation for non-block systems is very
          // slow and should be deemed as, mainly, a debug or prototyping
          // feature.
          if (static_cast<bool>(preconditioner.underlying_entity() &
                                Preconditioners::UnderlyingEntity::System))
            {
              if (nonlinear_params.fdm_precond_system_approximation)
                {
                  nonlinear_operator.initialize_system_matrix(false);
                  auto &system_matrix = nonlinear_operator.get_system_matrix();

                  calc_numeric_tangent(dof_handler,
                                       nonlinear_operator,
                                       current_u,
                                       nl_residual,
                                       system_matrix);
                }
              else
                {
                  nonlinear_operator.initialize_system_matrix(true);
                }
            }

          if (static_cast<bool>(preconditioner.underlying_entity() &
                                Preconditioners::UnderlyingEntity::BlockSystem))
            nonlinear_operator.initialize_block_system_matrix(true);

          preconditioner.do_update();
        };

      // Lambda to solve system using jacobian
      const auto nl_solve_with_jacobian = [&](const auto &src, auto &dst) {
        ScopedName sc("solve_with_jacobian");
        MyScope    scope(timer, sc);

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.set_zero(const_cast<VectorType &>(src).block(b));

        const unsigned int n_iterations = linear_solver->solve(dst, src);

        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.distribute(dst.block(b));

        if (nonlinear_params.verbosity >= 2 &&
            !solver_control_l->get_history_data().empty())
          {
            // DEBUG - temporary disabled
            /*
            pcout << " - l_res_abs: ";
            for (const auto res : solver_control_l->get_history_data())
              pcout << res << " ";
            pcout << std::endl;
            */

            std::vector<double> &res_history =
              const_cast<std::vector<double> &>(
                solver_control_l->get_history_data());
            res_history.clear();
          }

        statistics.increment_linear_iterations(n_iterations);

        return n_iterations;
      };

      if (nonlinear_params.nonlinear_solver_type == "damped")
        {
          DampedNewtonSolver<VectorType> non_linear_solver(
            statistics,
            NewtonSolverAdditionalData(
              nonlinear_params.newton_do_update,
              nonlinear_params.newton_threshold_newton_iter,
              nonlinear_params.newton_threshold_linear_iter,
              nonlinear_params.newton_reuse_preconditioner,
              nonlinear_params.newton_use_damping));

          non_linear_solver.reinit_vector = [&](auto &vector) {
            ScopedName sc("reinit_vector");
            MyScope    scope(timer, sc);

            nonlinear_operator.initialize_dof_vector(vector);
          };

          non_linear_solver.residual            = nl_residual;
          non_linear_solver.setup_jacobian      = nl_setup_jacobian;
          non_linear_solver.solve_with_jacobian = nl_solve_with_jacobian;

          if (nonlinear_params.l_solver != "Direct")
            non_linear_solver.setup_preconditioner = nl_setup_preconditioner;

          if (nonlinear_params.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status = check_iteration_status;

          non_linear_solver_executor = std::make_unique<
            NonLinearSolverWrapper<VectorType, DampedNewtonSolver<VectorType>>>(
            std::move(non_linear_solver));
        }
      else if (nonlinear_params.nonlinear_solver_type == "NOX")
        {
          Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
            Teuchos::rcp(new Teuchos::ParameterList);

          non_linear_parameters->set("Nonlinear Solver", "Line Search Based");

          auto &printParams = non_linear_parameters->sublist("Printing");
          printParams.set("Output Information",
                          nonlinear_params.nox_data.output_information);

          auto &dir_parameters = non_linear_parameters->sublist("Direction");
          dir_parameters.set("Method",
                             nonlinear_params.nox_data.direction_method);

          /* Disable the recovery feature of NOX such that it behaves similarly
           * to our damped solver in the case of linear solver failure. It has
           * been observed that recovery steps are not efficient for our
           * problem, so no need to waste time on them. */
          dir_parameters.sublist("Newton").set("Rescue Bad Newton Solve",
                                               false);

          auto &search_parameters =
            non_linear_parameters->sublist("Line Search");
          search_parameters.set("Method",
                                nonlinear_params.nox_data.line_search_method);

          // Params for polynomial
          auto &poly_params = search_parameters.sublist("Polynomial");
          poly_params.set(
            "Interpolation Type",
            nonlinear_params.nox_data.line_search_interpolation_type);

          typename TrilinosWrappers::NOXSolver<VectorType>::AdditionalData
            additional_data(nonlinear_params.nl_max_iter,
                            nonlinear_params.nl_abs_tol,
                            nonlinear_params.nl_rel_tol,
                            nonlinear_params.newton_threshold_newton_iter,
                            nonlinear_params.newton_threshold_linear_iter,
                            nonlinear_params.newton_reuse_preconditioner);

          TrilinosWrappers::NOXSolver<VectorType> non_linear_solver(
            additional_data, non_linear_parameters);

          non_linear_solver.residual = [&nl_residual](const auto &src,
                                                      auto       &dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [&nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          if (nonlinear_params.l_solver != "Direct")
            non_linear_solver.setup_preconditioner =
              [&nl_setup_preconditioner](const auto &current_u) {
                nl_setup_preconditioner(current_u);
                return 0;
              };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [&nl_solve_with_jacobian](const auto  &src,
                                      auto        &dst,
                                      const double tolerance) {
              (void)tolerance;
              return nl_solve_with_jacobian(src, dst);
            };

          non_linear_solver.apply_jacobian = [&j_op](const auto &src,
                                                     auto       &dst) {
            j_op.vmult(dst, src);
            return 0;
          };

          if (nonlinear_params.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status = check_iteration_status;

          non_linear_solver_executor = std::make_unique<
            NonLinearSolverWrapper<VectorType,
                                   TrilinosWrappers::NOXSolver<VectorType>>>(
            std::move(non_linear_solver), statistics);
        }
#if defined(DEAL_II_WITH_PETSC) && defined(USE_SNES)
      else if (nonlinear_params.nonlinear_solver_type == "SNES")
        {
          typename SNESSolver<VectorType>::AdditionalData additional_data(
            nonlinear_params.nl_max_iter,
            nonlinear_params.nl_abs_tol,
            nonlinear_params.nl_rel_tol,
            nonlinear_params.newton_threshold_newton_iter,
            nonlinear_params.newton_threshold_linear_iter,
            nonlinear_params.snes_data.line_search_name);

          SNESSolver<VectorType> non_linear_solver(
            additional_data, nonlinear_params.snes_data.solver_name);

          non_linear_solver.residual = [&nl_residual](const auto &src,
                                                      auto       &dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [&nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          non_linear_solver.setup_preconditioner =
            [&nl_setup_preconditioner](const auto &current_u) {
              nl_setup_preconditioner(current_u);
              return 0;
            };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [&nl_solve_with_jacobian](const auto  &src,
                                      auto        &dst,
                                      const double tolerance) {
              (void)tolerance;
              return nl_solve_with_jacobian(src, dst);
            };

          non_linear_solver.apply_jacobian = [&j_op](const auto &src,
                                                     auto       &dst) {
            j_op.vmult(dst, src);
            return 0;
          };

          if (nonlinear_params.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status = check_iteration_status;

          non_linear_solver_executor = std::make_unique<
            NonLinearSolverWrapper<VectorType, SNESSolver<VectorType>>>(
            std::move(non_linear_solver), statistics);
        }
#endif
      else
        AssertThrow(false, ExcNotImplemented());
    }

    void
    make_step(VectorType &current_solution) const override
    {
      non_linear_solver_executor->solve(current_solution);
    }

    void
    clear() const override
    {
      non_linear_solver_executor->clear();
    }

  private:
    std::unique_ptr<SolverControl> solver_control_l;

    std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

    std::unique_ptr<JacobianBase<Number>> jacobian_operator;

    std::unique_ptr<NewtonSolver<VectorType>> non_linear_solver_executor;
  };

} // namespace TimeIntegration