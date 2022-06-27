#pragma once

#include <pf-applications/base/timer.h>

namespace NonLinearSolvers
{
  using namespace dealii;

  struct NewtonSolverSolverControl
  {
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

    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;

    unsigned int newton_iterations    = 0;
    unsigned int linear_iterations    = 0;
    unsigned int residual_evaluations = 0;
  };



  DeclException1(
    ExcNewtonDidNotConverge,
    std::string,
    << "Damped Newton iteration did not converge. Maximum number of " << arg1
    << " iterations exceed!");


  struct NewtonSolverAdditionalData
  {
    NewtonSolverAdditionalData(const bool         do_update             = true,
                               const unsigned int threshold_newton_iter = 10,
                               const unsigned int threshold_linear_iter = 20,
                               const bool         reuse_preconditioner  = true)
      : do_update(do_update)
      , threshold_newton_iter(threshold_newton_iter)
      , threshold_linear_iter(threshold_linear_iter)
      , reuse_preconditioner(reuse_preconditioner)
    {}

    const bool         do_update;
    const unsigned int threshold_newton_iter;
    const unsigned int threshold_linear_iter;
    const bool         reuse_preconditioner;
  };



  template <typename VectorType>
  class NewtonSolver
  {
  public:
    NewtonSolver(NewtonSolverSolverControl &       statistics,
                 const NewtonSolverAdditionalData &solver_data_in =
                   NewtonSolverAdditionalData())
      : statistics(statistics)
      , solver_data(solver_data_in)
    {}

    void
    solve(VectorType &dst) const
    {
      VectorType vec_residual, increment, tmp;
      reinit_vector(vec_residual);
      reinit_vector(increment);
      reinit_vector(tmp);

      if (this->solver_data.reuse_preconditioner == false)
        clear();

      // Accumulated linear iterations
      this->statistics.clear();

      // evaluate residual using the given estimate of the solution
      residual(dst, vec_residual);
      ++statistics.residual_evaluations;

      double norm_r   = vec_residual.l2_norm();
      double norm_r_0 = norm_r;

      while (norm_r > this->statistics.abs_tol &&
             norm_r / norm_r_0 > statistics.rel_tol &&
             statistics.newton_iterations < statistics.max_iter)
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

          setup_jacobian(dst, solver_data.do_update && threshold_exceeded);

          history_linear_iterations_last =
            solve_with_jacobian(vec_residual, increment);

          statistics.linear_iterations += history_linear_iterations_last;

          // damped Newton scheme
          const double tau =
            0.1; // another parameter (has to be smaller than 1)
          const unsigned int N_ITER_TMP_MAX =
            100;                   // iteration counts for damping scheme
          double omega      = 1.0; // damping factor
          double norm_r_tmp = 1.0; // norm of residual using temporary solution
          unsigned int n_iter_tmp = 0;

          do
            {
              // calculate temporary solution
              tmp = dst;
              tmp.add(omega, increment);

              // evaluate residual using the temporary solution
              residual(tmp, vec_residual);
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

          // increment iteration counter
          ++statistics.newton_iterations;
          ++history_newton_iterations;
        }

      AssertThrow(norm_r <= this->statistics.abs_tol ||
                    norm_r / norm_r_0 <= statistics.rel_tol,
                  ExcNewtonDidNotConverge("Newton"));
    }

    void
    clear() const
    {
      history_linear_iterations_last = 0;
      history_newton_iterations      = 0;
    }


  private:
    NewtonSolverSolverControl &      statistics;
    const NewtonSolverAdditionalData solver_data;

    mutable unsigned int history_linear_iterations_last = 0;
    mutable unsigned int history_newton_iterations      = 0;

  public:
    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
  };
} // namespace NonLinearSolvers
