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

  NonLinearSolvers::NewtonSolverSolverControl solver_control;

  NonLinearSolvers::NOXSolver<VectorType> non_linear_solver(solver_control);
}