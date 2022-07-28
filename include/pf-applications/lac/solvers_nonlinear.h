#pragma once

#include <pf-applications/base/timer.h>

namespace NonLinearSolvers
{
  using namespace dealii;

  template <typename VectorType>
  class NewtonSolver;



  struct NewtonSolverSolverControl
  {
  public:
    enum State
    {
      iterate,
      success,
      failure
    };

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

    template <typename VectorType>
    State
    check(const unsigned int step,
          const double       check_value,
          const VectorType & solution,
          const VectorType & residuum)
    {
      (void)solution;
      (void)residuum;

      if (step == 0)
        this->check_value_0 = check_value;

      this->check_value = check_value;

      this->newton_iterations = step;

      if (check_value > abs_tol && check_value / check_value_0 > rel_tol &&
          newton_iterations < max_iter)
        return iterate;

      if (check_value <= abs_tol || check_value / check_value_0 <= rel_tol)
        return success;

      return failure;
    }

    State
    check()
    {
      Vector<double> dummy;

      return check(this->newton_iterations, this->check_value, dummy, dummy);
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

    template <typename>
    friend class NewtonSolver;

    template <typename>
    friend class DampedNewtonSolver;
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
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
    std::function<NewtonSolverSolverControl::State(const unsigned int,
                                                   const double,
                                                   const VectorType &,
                                                   const VectorType &)>
      check_iteration_status = {};
  };



  template <typename VectorType>
  class DampedNewtonSolver : public NewtonSolver<VectorType>
  {
  public:
    DampedNewtonSolver(NewtonSolverSolverControl &       statistics,
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
      ++statistics.residual_evaluations;

      double   norm_r = vec_residual.l2_norm();
      unsigned it     = 0;

      auto status = NewtonSolverSolverControl::iterate;

      while (status == NewtonSolverSolverControl::iterate)
        {
          status = check(it, norm_r, dst, vec_residual);

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

          this->setup_jacobian(dst,
                               solver_data.do_update && threshold_exceeded);

          history_linear_iterations_last =
            this->solve_with_jacobian(vec_residual, increment);

          statistics.linear_iterations += history_linear_iterations_last;

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
                  ++statistics.residual_evaluations;

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
              ++statistics.residual_evaluations;

              // calculate norm of residual
              norm_r = vec_residual.l2_norm();
            }

          // increment iteration counter
          ++it;
          ++history_newton_iterations;
        }

      AssertThrow(status == NewtonSolverSolverControl::success,
                  ExcNewtonDidNotConverge("Newton"));
    }

    void
    clear() const override
    {
      history_linear_iterations_last = 0;
      history_newton_iterations      = 0;
    }

  private:
    NewtonSolverSolverControl::State
    check(const unsigned int step,
          const double       check_value,
          const VectorType & x,
          const VectorType & r) const
    {
      if (this->check_iteration_status == nullptr)
        return statistics.check(step, check_value, x, r);

      const auto state1 = statistics.check(step, check_value, x, r);
      const auto state2 = this->check_iteration_status(step, check_value, x, r);

      if ((state1 == NewtonSolverSolverControl::failure) ||
          (state2 == NewtonSolverSolverControl::failure))
        return NewtonSolverSolverControl::failure;
      else if ((state1 == NewtonSolverSolverControl::iterate) ||
               (state2 == NewtonSolverSolverControl::iterate))
        return NewtonSolverSolverControl::iterate;
      else
        return NewtonSolverSolverControl::success;
    }

    NewtonSolverSolverControl &      statistics;
    const NewtonSolverAdditionalData solver_data;

    mutable unsigned int history_linear_iterations_last = 0;
    mutable unsigned int history_newton_iterations      = 0;
  };
} // namespace NonLinearSolvers
