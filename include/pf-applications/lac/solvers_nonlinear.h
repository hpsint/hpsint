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

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <deal.II/trilinos/nox.h>

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/jacobian_operator.h>
#include <pf-applications/lac/solvers_nonlinear_snes.h>

namespace NonLinearSolvers
{
  using namespace dealii;

  template <typename VectorType>
  class NewtonSolver;


  struct NewtonSolverSolverControl
  {
  public:
    NewtonSolverSolverControl(const unsigned int max_iter = 10,
                              const double       abs_tol  = 1.e-20,
                              const double       rel_tol  = 1.e-5)
      : max_iter(max_iter)
      , abs_tol(abs_tol)
      , rel_tol(rel_tol)
    {}

    void
    clear()
    {
      newton_iterations    = 0;
      linear_iterations    = 0;
      residual_evaluations = 0;
    }

    unsigned int
    n_newton_iterations() const
    {
      return newton_iterations;
    }

    unsigned int
    n_linear_iterations() const
    {
      return linear_iterations;
    }

    unsigned int
    n_residual_evaluations() const
    {
      return residual_evaluations;
    }

    void
    increment_newton_iterations(const unsigned int num)
    {
      newton_iterations += num;
    }

    void
    increment_linear_iterations(const unsigned int num)
    {
      linear_iterations += num;
    }

    void
    increment_residual_evaluations(const unsigned int num)
    {
      residual_evaluations += num;
    }

    template <typename VectorType>
    SolverControl::State
    check(const unsigned int step,
          const double       check_value,
          const VectorType  &solution,
          const VectorType  &residuum)
    {
      (void)solution;
      (void)residuum;

      if (step == 0)
        this->check_value_0 = check_value;

      this->check_value = check_value;

      this->newton_iterations = step;

      if (check_value > abs_tol && check_value / check_value_0 > rel_tol &&
          newton_iterations < max_iter)
        return SolverControl::iterate;

      if (check_value <= abs_tol || check_value / check_value_0 <= rel_tol)
        return SolverControl::success;

      return SolverControl::failure;
    }

    SolverControl::State
    check()
    {
      Vector<double> dummy;

      return check(this->newton_iterations, this->check_value, dummy, dummy);
    }

    unsigned int
    get_max_iter() const
    {
      return max_iter;
    }

    double
    get_abs_tol() const
    {
      return abs_tol;
    }

    double
    get_rel_tol() const
    {
      return rel_tol;
    }

  private:
    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;

    double check_value_0 = 0.0;
    double check_value   = 0.0;

    unsigned int newton_iterations    = 0;
    unsigned int linear_iterations    = 0;
    unsigned int residual_evaluations = 0;
  };



  struct NewtonSolverAdditionalData
  {
    NewtonSolverAdditionalData(const bool         do_update             = true,
                               const unsigned int threshold_newton_iter = 10,
                               const unsigned int threshold_linear_iter = 20,
                               const bool         reuse_preconditioner  = true,
                               const bool         use_damping           = true)
      : do_update(do_update)
      , threshold_newton_iter(threshold_newton_iter)
      , threshold_linear_iter(threshold_linear_iter)
      , reuse_preconditioner(reuse_preconditioner)
      , use_damping(use_damping)
    {}

    const bool         do_update;
    const unsigned int threshold_newton_iter;
    const unsigned int threshold_linear_iter;
    const bool         reuse_preconditioner;
    const bool         use_damping;
  };



  class ExcNewtonDidNotConverge : public dealii::ExceptionBase
  {
  public:
    ExcNewtonDidNotConverge(const std::string &label)
      : label(label)
    {}

    virtual ~ExcNewtonDidNotConverge() noexcept override = default;

    virtual void
    print_info(std::ostream &out) const override
    {
      out << message() << std::endl;
    }

    std::string
    message() const
    {
      return "Maximum number of " + this->label +
             " iterations of damp Netwon exceeded!";
    }

  private:
    const std::string label;
  };



  template <typename VectorType>
  class NewtonSolver
  {
  public:
    virtual void
    solve(VectorType &dst) const = 0;

    virtual void
    clear() const = 0;

    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &)>               setup_jacobian = {};
    std::function<void(const VectorType &)> setup_preconditioner         = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
    std::function<SolverControl::State(const unsigned int,
                                       const double,
                                       const VectorType &,
                                       const VectorType &)>
      check_iteration_status = {};
  };



  template <typename VectorType>
  class DampedNewtonSolver : public NewtonSolver<VectorType>
  {
  public:
    DampedNewtonSolver(NewtonSolverSolverControl        &statistics,
                       const NewtonSolverAdditionalData &solver_data_in =
                         NewtonSolverAdditionalData())
      : statistics(statistics)
      , solver_data(solver_data_in)
    {}

    void
    solve(VectorType &dst) const override
    {
      VectorType vec_residual, increment, tmp;
      this->reinit_vector(vec_residual);
      this->reinit_vector(increment);
      this->reinit_vector(tmp);

      if (this->solver_data.reuse_preconditioner == false)
        clear();

      // Accumulated linear iterations
      this->statistics.clear();

      // evaluate residual using the given estimate of the solution
      this->residual(dst, vec_residual);

      double   norm_r = vec_residual.l2_norm();
      unsigned it     = 0;

      auto status = check(it, norm_r, dst, vec_residual);

      while (status == SolverControl::iterate)
        {
          // reset increment
          increment = 0.0;

          // multiply by -1.0 since the linearized problem is "LinearMatrix *
          // increment = - vec_residual"
          vec_residual *= -1.0;

          // solve linear problem
          const bool threshold_exceeded =
            (history_newton_iterations % solver_data.threshold_newton_iter ==
             0) ||
            (history_linear_iterations_last >
             solver_data.threshold_linear_iter);

          this->setup_jacobian(dst);

          if (this->setup_preconditioner && solver_data.do_update &&
              threshold_exceeded)
            this->setup_preconditioner(dst);

          history_linear_iterations_last =
            this->solve_with_jacobian(vec_residual, increment);

          if (this->solver_data.use_damping)
            {
              // damped Newton scheme
              const double tau =
                0.1; // another parameter (has to be smaller than 1)
              const unsigned int N_ITER_TMP_MAX =
                100;              // iteration counts for damping scheme
              double omega = 1.0; // damping factor
              double norm_r_tmp =
                1.0; // norm of residual using temporary solution
              unsigned int n_iter_tmp = 0;

              do
                {
                  // calculate temporary solution
                  tmp = dst;
                  tmp.add(omega, increment);

                  // evaluate residual using the temporary solution
                  this->residual(tmp, vec_residual);

                  // calculate norm of residual (for temporary solution)
                  norm_r_tmp = vec_residual.l2_norm();

                  // reduce step length
                  omega = omega / 2.0;

                  // increment counter
                  n_iter_tmp++;
                }
              while (norm_r_tmp >= (1.0 - tau * omega) * norm_r &&
                     n_iter_tmp < N_ITER_TMP_MAX);

              AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                          ExcNewtonDidNotConverge("damping"));

              // update solution and residual
              dst    = tmp;
              norm_r = norm_r_tmp;
            }
          else
            {
              // non-damped Newton scheme

              // calculate temporary solution
              dst.add(1.0, increment);

              // evaluate residual
              this->residual(dst, vec_residual);

              // calculate norm of residual
              norm_r = vec_residual.l2_norm();
            }

          // increment iteration counter
          ++it;
          ++history_newton_iterations;

          status = check(it, norm_r, dst, vec_residual);
        }

      AssertThrow(status == SolverControl::success,
                  ExcNewtonDidNotConverge("Newton"));
    }

    void
    clear() const override
    {
      history_linear_iterations_last = 0;
      history_newton_iterations      = 0;
    }

  private:
    SolverControl::State
    check(const unsigned int step,
          const double       check_value,
          const VectorType  &x,
          const VectorType  &r) const
    {
      if (this->check_iteration_status == nullptr)
        return statistics.check(step, check_value, x, r);

      const auto state1 = statistics.check(step, check_value, x, r);
      const auto state2 = this->check_iteration_status(step, check_value, x, r);

      if ((state1 == SolverControl::failure) ||
          (state2 == SolverControl::failure))
        return SolverControl::failure;
      else if ((state1 == SolverControl::iterate) ||
               (state2 == SolverControl::iterate))
        return SolverControl::iterate;
      else
        return SolverControl::success;
    }

    NewtonSolverSolverControl       &statistics;
    const NewtonSolverAdditionalData solver_data;

    mutable unsigned int history_linear_iterations_last = 0;
    mutable unsigned int history_newton_iterations      = 0;
  };

  template <typename VectorType, typename SolverType>
  class NonLinearSolverWrapper : public NewtonSolver<VectorType>
  {
  public:
    NonLinearSolverWrapper(SolverType &&solver)
      : solver(std::move(solver))
    {}

    void
    clear() const override
    {
      solver.clear();
    }

    void
    solve(VectorType &dst) const override
    {
      solver.solve(dst);
    }

  private:
    mutable SolverType solver;
  };

  template <typename VectorType>
  class NonLinearSolverWrapper<VectorType,
                               TrilinosWrappers::NOXSolver<VectorType>>
    : public NewtonSolver<VectorType>
  {
  public:
    NonLinearSolverWrapper(TrilinosWrappers::NOXSolver<VectorType> &&solver,
                           NewtonSolverSolverControl                &statistics)
      : solver(std::move(solver))
      , statistics(statistics)
    {}

    void
    clear() const override
    {
      solver.clear();
    }

    void
    solve(VectorType &dst) const override
    {
      try
        {
          const unsigned int n_newton_iterations = solver.solve(dst);
          statistics.increment_newton_iterations(n_newton_iterations);
        }
      catch (const TrilinosWrappers::ExcNOXNoConvergence &e)
        {
          AssertThrow(false, ExcNewtonDidNotConverge("NOX"));
        }
    }

  private:
    mutable TrilinosWrappers::NOXSolver<VectorType> solver;

    NewtonSolverSolverControl &statistics;
  };

#if defined(DEAL_II_WITH_PETSC) && defined(USE_SNES)

  template <typename VectorType>
  class NonLinearSolverWrapper<VectorType, SNESSolver<VectorType>>
    : public NewtonSolver<VectorType>
  {
  public:
    NonLinearSolverWrapper(SNESSolver<VectorType>   &&solver,
                           NewtonSolverSolverControl &statistics)
      : solver(std::move(solver))
      , statistics(statistics)
    {}

    void
    clear() const override
    {
      solver.clear();
    }

    void
    solve(VectorType &dst) const override
    {
      try
        {
          const unsigned int n_newton_iterations = solver.solve(dst);
          statistics.increment_newton_iterations(n_newton_iterations);
        }
      catch (const ExcPETScError &e)
        {
          AssertThrow(false, ExcNewtonDidNotConverge("SNES"));
        }
      catch (const ExcSNESNoConvergence &e)
        {
          AssertThrow(false, ExcNewtonDidNotConverge("SNES"));
        }
    }

  private:
    mutable SNESSolver<VectorType> solver;

    NewtonSolverSolverControl &statistics;
  };

#endif
} // namespace NonLinearSolvers
