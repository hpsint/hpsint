#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/solvers_nonlinear.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // set up solver control
  const unsigned int n_max_iterations = 100;
  const double       abs_tolerance    = 1e-9;
  const double       rel_tolerance    = 1e-3;

  NonLinearSolvers::NewtonSolverSolverControl statistics(n_max_iterations,
                                                         abs_tolerance,
                                                         rel_tolerance);

  // set up parameters
  Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
    Teuchos::rcp(new Teuchos::ParameterList);

  non_linear_parameters->set("Nonlinear Solver", "Line Search Based");

  auto &dir_parameters = non_linear_parameters->sublist("Direction");
  dir_parameters.set("Method", "Newton");

  auto &search_parameters = non_linear_parameters->sublist("Line Search");
  search_parameters.set("Method", "Polynomial");

  // set up solver
  NonLinearSolvers::NOXSolver<VectorType> solver(statistics,
                                                 non_linear_parameters);

  // ... helper functions
  double J = 0.0;

  solver.residual = [](const auto &src, auto &dst) {
    // compute residual
    dst[0] = src[0] * src[0];
  };

  solver.setup_jacobian = [&](const auto &src, const auto) {
    // compute Jacobian
    J = 2.0 * src[0];
  };

  solver.solve_with_jacobian = [&](const auto &src, auto &dst) {
    // solve with Jacobian
    dst[0] = src[0] / J;

    return 1;
  };

  // initial guess
  VectorType solution(1);
  solution[0] = 2.0;

  // solve with the given initial guess
  solver.solve(solution);
}