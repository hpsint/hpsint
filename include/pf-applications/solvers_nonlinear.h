#pragma once

#include <pf-applications/timer.h>

namespace NonLinearSolvers
{
  using namespace dealii;

  struct NonLinearSolverStatistics
  {
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;
  };



  DeclExceptionMsg(
    ExcNewtonDidNotConverge,
    "Damped Newton iteration did not converge. Maximum number of iterations exceed!");



  struct NewtonSolverData
  {
    NewtonSolverData(const unsigned int max_iter              = 100,
                     const double       abs_tol               = 1.e-20,
                     const double       rel_tol               = 1.e-5,
                     const bool         do_update             = true,
                     const unsigned int threshold_newton_iter = 10,
                     const unsigned int threshold_linear_iter = 20)
      : max_iter(max_iter)
      , abs_tol(abs_tol)
      , rel_tol(rel_tol)
      , do_update(do_update)
      , threshold_newton_iter(threshold_newton_iter)
      , threshold_linear_iter(threshold_linear_iter)
    {}

    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;
    const bool         do_update;
    const unsigned int threshold_newton_iter;
    const unsigned int threshold_linear_iter;
  };



  template <typename VectorType>
  class NewtonSolver
  {
  public:
    NewtonSolver(const NewtonSolverData &solver_data_in = NewtonSolverData())
      : solver_data(solver_data_in)
    {}

    NonLinearSolverStatistics
    solve(VectorType &dst) const
    {
      VectorType vec_residual, increment, tmp;
      reinit_vector(vec_residual);
      reinit_vector(increment);
      reinit_vector(tmp);

      // evaluate residual using the given estimate of the solution
      residual(dst, vec_residual);

      double norm_r   = vec_residual.l2_norm();
      double norm_r_0 = norm_r;

      // Accumulated linear iterations
      NonLinearSolverStatistics statistics;

      unsigned int linear_iterations_last = 0;

      while (norm_r > this->solver_data.abs_tol &&
             norm_r / norm_r_0 > solver_data.rel_tol &&
             statistics.newton_iterations < solver_data.max_iter)
        {
          // reset increment
          increment = 0.0;

          // multiply by -1.0 since the linearized problem is "LinearMatrix *
          // increment = - vec_residual"
          vec_residual *= -1.0;

          // solve linear problem
          bool const threshold_exceeded =
            (statistics.newton_iterations % solver_data.threshold_newton_iter ==
             0) ||
            (linear_iterations_last > solver_data.threshold_linear_iter);

          setup_jacobian(dst, solver_data.do_update && threshold_exceeded);

          linear_iterations_last = solve_with_jacobian(vec_residual, increment);

          statistics.linear_iterations += linear_iterations_last;

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
                      ExcNewtonDidNotConverge());

          // update solution and residual
          dst    = tmp;
          norm_r = norm_r_tmp;

          // increment iteration counter
          ++statistics.newton_iterations;
        }

      AssertThrow(norm_r <= this->solver_data.abs_tol ||
                    norm_r / norm_r_0 <= solver_data.rel_tol,
                  ExcNewtonDidNotConverge());

      return statistics;
    }


  private:
    const NewtonSolverData solver_data;

  public:
    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
  };
} // namespace NonLinearSolvers
