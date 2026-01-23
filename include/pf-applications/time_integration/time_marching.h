// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 by the hpsint authors
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

#include <pf-applications/base/data.h>
#include <pf-applications/base/scoped_name.h>

#include <pf-applications/lac/evaluation.h>
#include <pf-applications/lac/preconditioners.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_linear_tools.h>
#include <pf-applications/lac/solvers_nonlinear.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

#include <pf-applications/pde/equation_type.h>
#include <pf-applications/time_integration/time_schemes.h>

namespace TimeIntegration
{
  template <typename VectorType>
  class TimeMarching
  {
  public:
    virtual ~TimeMarching() = default;

    virtual void
    make_step(VectorType &) = 0;

    virtual void
    clear() = 0;
  };

  using namespace dealii;
  using namespace hpsint;
  using namespace NonLinearSolvers;

  template <int dim,
            typename VectorType,
            typename NonLinearOperator,
            typename StreamType>
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
      StreamType                                  &stream,
      std::function<void(const VectorType &)>      setup_custom_preconditioner,
      std::function<void(const VectorType &)>      setup_linearization_point,
      std::function<
        std::vector<std::tuple<std::string, unsigned int, unsigned int>>(void)>
                 quantities_to_check,
      const bool print_stats = true)
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
        [&, slp = std::move(setup_linearization_point), nl_residual](
          const auto &current_u) {
          ScopedName sc("setup_jacobian");
          MyScope    scope(timer, sc);

          slp(current_u);

          // TODO disable this feature for a while, fix later
          // nonlinear_operator.update_state(current_u);

          nonlinear_operator.do_update();

          if (nonlinear_params.fdm_jacobian_approximation)
            {
              nonlinear_operator.initialize_system_matrix(false, print_stats);
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
        [&, scp = std::move(setup_custom_preconditioner), nl_residual](
          const auto &current_u) {
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
                  nonlinear_operator.initialize_system_matrix(false,
                                                              print_stats);
                  auto &system_matrix = nonlinear_operator.get_system_matrix();

                  calc_numeric_tangent(dof_handler,
                                       nonlinear_operator,
                                       current_u,
                                       nl_residual,
                                       system_matrix);
                }
              else
                {
                  nonlinear_operator.initialize_system_matrix(true,
                                                              print_stats);
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
            stream << " - l_res_abs: ";
            for (const auto res : solver_control_l->get_history_data())
              stream << res << " ";
            stream << std::endl;

            std::vector<double> &res_history =
              const_cast<std::vector<double> &>(
                solver_control_l->get_history_data());
            res_history.clear();
          }

        statistics.increment_linear_iterations(n_iterations);

        return n_iterations;
      };

      const auto nl_check_iteration_status =
        [&,
         qtc                  = std::move(quantities_to_check),
         check_value_0        = 0.0,
         check_values_0       = std::vector<Number>(),
         previous_linear_iter = 0](const auto  step,
                                   const auto  check_value,
                                   const auto &x,
                                   const auto &r) mutable {
          (void)x;

          const auto check_qtys = qtc();

          std::vector<Number> check_values;
          for (const auto &qty : check_qtys)
            {
              const unsigned int offset       = std::get<1>(qty);
              const unsigned int n_components = std::get<2>(qty);

              Number val = 0;
              for (unsigned int i = 0; i < n_components; ++i)
                val += r.block(offset + i).norm_sqr();

              check_values.push_back(std::sqrt(val));
            }

          if (step == 0)
            {
              check_value_0        = check_value;
              check_values_0       = check_values;
              previous_linear_iter = 0;
            }

          const unsigned int step_linear_iter =
            statistics.n_linear_iterations() - previous_linear_iter;

          previous_linear_iter = statistics.n_linear_iterations();

          if (stream.is_active())
            {
              stream << std::scientific << std::setprecision(6);

              if (step == 0)
                {
                  stream << "\nit";
                  stream << std::setw(13) << std::right << "res_abs";
                  stream << std::setw(13) << std::right << "res_rel";

                  for (const auto &qty : check_qtys)
                    {
                      const std::string &name = std::get<0>(qty);
                      stream << std::setw(13) << std::right
                             << (name + "_res_abs");
                      stream << std::setw(13) << std::right
                             << (name + "_res_rel");
                    }

                  stream << std::setw(13) << std::right << "linear_iter"
                         << std::endl;
                }

              stream << std::setw(2) << std::right << step;
              stream << " " << check_value;
              if (step > 0)
                stream << " "
                       << (check_value_0 ? check_value / check_value_0 : 0.);
              else
                stream << " ------------";

              for (unsigned int i = 0; i < check_values.size(); ++i)
                {
                  stream << " " << check_values[i];

                  if (step > 0)
                    stream << " "
                           << (check_values_0[i] ?
                                 check_values[i] / check_values_0[i] :
                                 0.);
                  else
                    stream << " ------------";
                }
              stream << std::setw(13) << std::right << step_linear_iter
                     << std::endl;

              stream << std::defaultfloat;
            }

          /* This function does not really test anything and simply prints more
           * details on the residual evolution. We have different return status
           * here due to the fact that our DampedNewtonSolver::check() works
           * slightly differently in comparison to
           * NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR). The latter has
           * a bit strange not very obvious logic.
           */
          return nonlinear_params.nonlinear_solver_type == "NOX" ?
                   SolverControl::iterate :
                   SolverControl::success;
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
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

          non_linear_solver_executor = std::make_unique<
            NonLinearSolverWrapper<VectorType, DampedNewtonSolver<VectorType>>>(
            std::move(non_linear_solver));
        }
      else if (nonlinear_params.nonlinear_solver_type == "NOX")
        {
          Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
            Teuchos::rcp(new Teuchos::ParameterList);

          non_linear_parameters->set(
            "Nonlinear Solver", nonlinear_params.nox_data.nonlinear_solver);

          auto &print_params = non_linear_parameters->sublist("Printing");
          print_params.set("Output Information",
                           nonlinear_params.nox_data.output_information);

          auto &dir_parameters = non_linear_parameters->sublist("Direction");
          dir_parameters.set("Method",
                             nonlinear_params.nox_data.direction_method);

          auto &newton_params = dir_parameters.sublist("Newton");

          /* Disable the recovery feature of NOX such that it behaves similarly
           * to our damped solver in the case of linear solver failure. It has
           * been observed that recovery steps are not efficient for our
           * problem, so no need to waste time on them. */
          newton_params.set("Rescue Bad Newton Solve", false);

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

          non_linear_solver.residual = [nl_residual](const auto &src,
                                                     auto       &dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          if (nonlinear_params.l_solver != "Direct")
            non_linear_solver.setup_preconditioner =
              [nl_setup_preconditioner](const auto &current_u) {
                nl_setup_preconditioner(current_u);
                return 0;
              };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [nl_solve_with_jacobian](const auto  &src,
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
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

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

          non_linear_solver.residual = [nl_residual](const auto &src,
                                                     auto       &dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          non_linear_solver.setup_preconditioner =
            [nl_setup_preconditioner](const auto &current_u) {
              nl_setup_preconditioner(current_u);
              return 0;
            };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [nl_solve_with_jacobian](const auto  &src,
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
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

          non_linear_solver_executor = std::make_unique<
            NonLinearSolverWrapper<VectorType, SNESSolver<VectorType>>>(
            std::move(non_linear_solver), statistics);
        }
#endif
      else
        AssertThrow(false, ExcNotImplemented());
    }

    void
    make_step(VectorType &current_solution) override
    {
      non_linear_solver_executor->solve(current_solution);
    }

    void
    clear() override
    {
      non_linear_solver_executor->clear();
    }

  private:
    std::unique_ptr<SolverControl> solver_control_l;

    std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

    std::unique_ptr<JacobianBase<Number>> jacobian_operator;

    std::unique_ptr<NewtonSolver<VectorType>> non_linear_solver_executor;
  };

  template <typename VectorType,
            typename NonLinearOperator,
            typename StreamType>
  class TimeMarchingExplicit : public TimeMarching<VectorType>
  {
  public:
    using Number = typename NonLinearOperator::value_type;

  private:
    template <typename Operator>
    using LinearSolverCG = LinearSolvers::
      SolverCGWrapper<Operator, Preconditioners::PreconditionerBase<Number>>;

  public:
    TimeMarchingExplicit(const NonLinearOperator         &nonlinear_operator,
                         const ResidualWrapper<Number>   &residual_wrapper,
                         const AffineConstraints<Number> &constraints,
                         std::unique_ptr<ExplicitScheme>  integration_scheme,
                         NewtonSolverSolverControl       &statistics,
                         MyTimerOutput                   &timer,
                         StreamType                      &stream)
      : nonlinear_operator(nonlinear_operator)
      , residual_wrapper(residual_wrapper)
      , constraints(constraints)
      , integration_scheme(std::move(integration_scheme))
      , statistics(statistics)
      , timer(timer)
      , stream(stream)
      , linear_solver_wrapper(nullptr)
    {
      // Check that operator knows what kind of equations it contains
      // Check if there are any stationary equations in the operator
      for (unsigned int c = 0; c < nonlinear_operator.n_components(); ++c)
        {
          AssertThrow(nonlinear_operator.equation_type(c) !=
                        EquationType::Undefined,
                      ExcMessage("Equation type is not set for component " +
                                 std::to_string(c)));

          has_stationary_equations =
            has_stationary_equations ||
            nonlinear_operator.equation_type(c) == EquationType::Stationary;
        }
    }

    template <typename MassOperator>
    TimeMarchingExplicit(const NonLinearOperator         &nonlinear_operator,
                         MassOperator                    &mass_operator,
                         const ResidualWrapper<Number>   &residual_wrapper,
                         const AffineConstraints<Number> &constraints,
                         std::unique_ptr<ExplicitScheme>  integration_scheme,
                         const NonLinearData             &nonlinear_params,
                         NewtonSolverSolverControl       &statistics,
                         MyTimerOutput                   &timer,
                         StreamType                      &stream)
      : TimeMarchingExplicit(nonlinear_operator,
                             residual_wrapper,
                             constraints,
                             std::move(integration_scheme),
                             statistics,
                             timer,
                             stream)
    {
      linear_solver_wrapper = make_unique_from(
        LinearSolvers::wrap_solver_with_preconditioner<MassOperator,
                                                       Preconditioners::IC,
                                                       LinearSolverCG>(
          mass_operator, nonlinear_params));

      setup_preconditioner = [this, &mass_operator]() {
        this->linear_solver_wrapper->preconditioner->clear();

        if (static_cast<bool>(
              this->linear_solver_wrapper->preconditioner->underlying_entity() &
              Preconditioners::UnderlyingEntity::System))
          mass_operator.initialize_system_matrix(true);
        else if (static_cast<bool>(
                   this->linear_solver_wrapper->preconditioner
                     ->underlying_entity() &
                   Preconditioners::UnderlyingEntity::BlockSystem))
          mass_operator.initialize_block_system_matrix(true);

        this->linear_solver_wrapper->preconditioner->do_update();
      };
    }

    void
    make_step(VectorType &current_solution) override
    {
      // We check only one vector since we reset them together
      if (increment.n_blocks() == 0)
        {
          nonlinear_operator.initialize_dof_vector(vec_k);
          nonlinear_operator.initialize_dof_vector(vec_p);
          nonlinear_operator.initialize_dof_vector(vec_k_minv);
          nonlinear_operator.initialize_dof_vector(increment);

          if (setup_preconditioner)
            setup_preconditioner();
        }

      // Get current timestep and time instance
      const auto dt = nonlinear_operator.get_data().time_data.get_current_dt();
      const auto t = nonlinear_operator.get_data().time_data.get_current_time();

      const auto &stages = integration_scheme->get_stages();

      // Copy initially if number of stages is more than 1
      if (stages.size() > 1)
        vec_p = current_solution;

      // If we have a single stage, we can work directly with the solution
      auto &vec_work = (stages.size() > 1) ? vec_p : current_solution;

      // Nullify incremet
      increment = 0;

      // Iterate over stages
      for (const auto &stage : stages)
        {
          const double stage_dt     = stage.first;
          const double stage_weight = stage.second;
          // Compute residual
          {
            ScopedName sc("residual");
            MyScope    scope(timer, sc);

            residual_wrapper.evaluate_nonlinear_residual(vec_k, vec_work);

            statistics.increment_residual_evaluations(1);
          }

          for (unsigned int b = 0; b < current_solution.n_blocks(); ++b)
            {
              const bool is_time_pde = nonlinear_operator.equation_type(b) ==
                                       EquationType::TimeDependent;

              if (linear_solver_wrapper)
                {
                  const auto view_lhs = is_time_pde ?
                                          vec_k_minv.create_view(b, b + 1) :
                                          vec_work.create_view(b, b + 1);
                  const auto view_rhs = vec_k.create_view(b, b + 1);

                  *view_lhs = 0;

                  constraints.set_zero(view_rhs->block(0));
                  const auto n_iterations =
                    linear_solver_wrapper->linear_solver->solve(*view_lhs,
                                                                *view_rhs);
                  constraints.distribute(view_lhs->block(0));

                  statistics.increment_linear_iterations(n_iterations);
                }
              else
                {
                  // No mass matrix provided, assume identity
                  constraints.distribute(vec_k.block(b));

                  if (is_time_pde)
                    vec_k_minv.block(b) = vec_k.block(b);
                  else
                    vec_work.block(b) = vec_k.block(b);
                }

              // Sign has to be inverted here, since operators are written (and
              // this is our convention) such that we solve by default equations
              // of form du/dt + f(u) = 0 or u + f(u) = 0, but in the explicit
              // time marching scheme we want to solve u = u + dt * (-f(u)) or u
              // = -f(u) instead.
              if (is_time_pde)
                {
                  vec_k_minv.block(b) *= -1.0 * stage_dt * dt;
                  increment.block(b) += vec_k_minv.block(b);

                  // Eventually it means we have n_stages > 1
                  if (stage_weight)
                    {
                      vec_k_minv.block(b) *= 1.0 / stage_dt * stage_weight;
                      vec_work.block(b) = current_solution.block(b);
                      vec_work.block(b) += vec_k_minv.block(b);
                    }
                }
              else
                {
                  vec_work.block(b) *= -1.0;
                }
            }

          // Set time for the next stage
          nonlinear_operator.get_data().time_data.set_current_time(
            t + stage_weight * dt);
        }

      // TODO: a better update should be implemented
      current_solution += increment;

      // We update the time step before updating the algebraic part
      nonlinear_operator.get_data().time_data.update_current_time(dt);

      // Separately update algebraic equations if any
      if (has_stationary_equations)
        {
          {
            ScopedName sc("residual");
            MyScope    scope(timer, sc);

            residual_wrapper.evaluate_nonlinear_residual(vec_k,
                                                         current_solution);

            statistics.increment_residual_evaluations(1);
          }

          for (unsigned int b = 0; b < current_solution.n_blocks(); ++b)
            {
              if (nonlinear_operator.equation_type(b) ==
                  EquationType::Stationary)
                {
                  if (linear_solver_wrapper)
                    {
                      const auto view_lhs =
                        current_solution.create_view(b, b + 1);
                      const auto view_rhs = vec_k.create_view(b, b + 1);

                      *view_lhs = 0;

                      constraints.set_zero(view_rhs->block(0));
                      const auto n_iterations =
                        linear_solver_wrapper->linear_solver->solve(*view_lhs,
                                                                    *view_rhs);
                      constraints.distribute(view_lhs->block(0));

                      statistics.increment_linear_iterations(n_iterations);
                    }
                  else
                    {
                      // No mass matrix provided, assume identity
                      current_solution.block(b) = vec_k.block(b);
                      constraints.distribute(current_solution.block(b));
                    }

                  current_solution.block(b) *= -1.0;
                }
            }
        }
    }

    void
    clear() override
    {
      // Clear the vectors
      vec_k.reinit(0);
      vec_k_minv.reinit(0);
      vec_p.reinit(0);
      increment.reinit(0);
    }

  private:
    const NonLinearOperator         &nonlinear_operator;
    const ResidualWrapper<Number>   &residual_wrapper;
    const AffineConstraints<Number> &constraints;

    const std::unique_ptr<ExplicitScheme> integration_scheme;

    NewtonSolverSolverControl &statistics;
    MyTimerOutput             &timer;
    StreamType                &stream;

    std::unique_ptr<LinearSolvers::LinearSolverWrapper<Number>>
      linear_solver_wrapper;

    std::function<void()> setup_preconditioner = {};

    VectorType vec_k;
    VectorType vec_k_minv;
    VectorType vec_p;
    VectorType increment;

    bool has_stationary_equations;
  };

} // namespace TimeIntegration