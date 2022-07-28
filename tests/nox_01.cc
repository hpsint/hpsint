#include <deal.II/base/config.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/solvers_nonlinear.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  VectorType solution;

  NonLinearSolvers::NewtonSolverSolverControl solver_control;

  NonLinearSolvers::NOXSolver<VectorType> non_linear_solver(solver_control);

  non_linear_solver.reinit_vector = [](auto &) {
    // TODO
  };

  non_linear_solver.residual = [](const auto &, auto &) {
    // TODO
  };

  non_linear_solver.setup_jacobian = [](const auto &, const auto) {
    // TODO
  };

  non_linear_solver.solve_with_jacobian = [](const auto &, auto &) {
    // TODO

    return 0;
  };

  non_linear_solver.check_iteration_status =
    [](const auto, const auto, auto &, auto &) {
      return NonLinearSolvers::NewtonSolverSolverControl::success;
    };

  non_linear_solver.solve(solution);

  non_linear_solver.clear();
}