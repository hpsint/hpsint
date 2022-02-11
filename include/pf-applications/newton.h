#pragma once

using namespace dealii;

//#define DEBUG_NORM

struct NewtonSolverData
{
  NewtonSolverData(const unsigned int max_iter = 100,
                   const double       abs_tol  = 1.e-20,
                   const double       rel_tol  = 1.e-5)
    : max_iter(max_iter)
    , abs_tol(abs_tol)
    , rel_tol(rel_tol)
  {}

  const unsigned int max_iter;
  const double       abs_tol;
  const double       rel_tol;
};



struct NewtonSolverStatistics
{
  unsigned int newton_iterations = 0;
  unsigned int linear_iterations = 0;
};



template <typename VectorType,
          typename NonlinearOperator,
          typename SolverLinearizedProblem>
class NewtonSolver
{
public:
  NewtonSolver(NonlinearOperator &      nonlinear_operator_in,
               SolverLinearizedProblem &linear_solver_in,
               const NewtonSolverData & solver_data_in = NewtonSolverData())
    : solver_data(solver_data_in)
    , nonlinear_operator(nonlinear_operator_in)
    , linear_solver(linear_solver_in)
  {
    nonlinear_operator.initialize_dof_vector(residual);
    nonlinear_operator.initialize_dof_vector(increment);
    nonlinear_operator.initialize_dof_vector(tmp);
  }

  NewtonSolverStatistics
  solve(VectorType &       dst,
        bool const         update_preconditioner_linear_solver     = true,
        unsigned int const update_preconditioner_every_newton_iter = true)
  {
    VectorType rhs;
    return this->solve(dst,
                       rhs,
                       update_preconditioner_linear_solver,
                       update_preconditioner_every_newton_iter);
  }



  NewtonSolverStatistics
  solve(VectorType &       dst,
        VectorType const & rhs,
        bool const         update_preconditioner_linear_solver     = true,
        unsigned int const update_preconditioner_every_newton_iter = true)
  {
    const bool constant_rhs = rhs.size() > 0;

    // evaluate residual using the given estimate of the solution
    nonlinear_operator.evaluate_nonlinear_residual(residual, dst);

    if (constant_rhs)
      residual -= rhs;

    double norm_r   = residual.l2_norm();
    double norm_r_0 = norm_r;

    // Accumulated linear iterations
    NewtonSolverStatistics statistics;

#ifdef DEBUG_NORM
    std::cout << "NORM: " << std::flush;
#endif

    while (norm_r > this->solver_data.abs_tol &&
           norm_r / norm_r_0 > solver_data.rel_tol &&
           statistics.newton_iterations < solver_data.max_iter)
      {
#ifdef DEBUG_NORM
        std::cout << norm_r << " " << std::flush;
#endif
        // reset increment
        increment = 0.0;

        // multiply by -1.0 since the linearized problem is "LinearMatrix *
        // increment = - residual"
        residual *= -1.0;

        // solve linear problem
        nonlinear_operator.set_solution_linearization(dst);
        nonlinear_operator.evaluate_newton_step(dst);
        bool const do_update = update_preconditioner_linear_solver &&
                               (statistics.newton_iterations %
                                  update_preconditioner_every_newton_iter ==
                                0);
        statistics.linear_iterations +=
          linear_solver.solve(increment, residual, do_update);

        // damped Newton scheme
        double omega      = 1.0; // damping factor
        double tau        = 0.1; // another parameter (has to be smaller than 1)
        double norm_r_tmp = 1.0; // norm of residual using temporary solution
        unsigned int n_iter_tmp = 0,
                     N_ITER_TMP_MAX =
                       100; // iteration counts for damping scheme

        do
          {
            // calculate temporary solution
            tmp = dst;
            tmp.add(omega, increment);


            // evaluate residual using the temporary solution
            nonlinear_operator.evaluate_nonlinear_residual(residual, tmp);
            if (constant_rhs)
              residual -= rhs;

            // calculate norm of residual (for temporary solution)
            norm_r_tmp = residual.l2_norm();

            // reduce step length
            omega = omega / 2.0;

            // increment counter
            n_iter_tmp++;
          }
        while (norm_r_tmp >= (1.0 - tau * omega) * norm_r &&
               n_iter_tmp < N_ITER_TMP_MAX);

        AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                    ExcMessage("Damped Newton iteration did not converge. "
                               "Maximum number of iterations exceeded!"));

        // update solution and residual
        dst    = tmp;
        norm_r = norm_r_tmp;

        // increment iteration counter
        ++statistics.newton_iterations;
      }

#ifdef DEBUG_NORM
    std::cout << std::endl;
#endif

    AssertThrow(
      norm_r <= this->solver_data.abs_tol ||
        norm_r / norm_r_0 <= solver_data.rel_tol,
      ExcMessage(
        "Newton solver failed to solve nonlinear problem to given tolerance. "
        "Maximum number of iterations exceeded!"));

    return statistics;
  }


private:
  NewtonSolverData         solver_data;
  NonlinearOperator &      nonlinear_operator;
  SolverLinearizedProblem &linear_solver;

  VectorType residual, increment, tmp;
};
