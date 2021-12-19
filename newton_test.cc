#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include "include/newton.h"



using namespace dealii;

template <int dim,
          int degree,
          int n_points_1D,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
class NonlinearOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  using value_type  = Number;
  using vector_type = VectorType;

  void
  initialize_dof_vector(VectorType &dst) const
  {
    (void)dst;
  }

  void
  set_solution_linearization(const VectorType &dst) const
  {
    (void)dst;
  }

  void
  evaluate_newton_step(const VectorType &newton_step)
  {
    (void)newton_step;
  }

  void
  evaluate_nonlinear_residual(const VectorType &src,
                              const VectorType &dst) const
  {
    (void)dst;
    (void)src;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    (void)dst;
    (void)src;
  }
};

template <typename Operator>
class SolverCGWrapper
{
public:
  using VectorType = typename Operator::vector_type;

  SolverCGWrapper(const Operator &op)
    : op(op)
  {}

  unsigned int
  solve(VectorType &dst, const VectorType &src, const bool do_update)
  {
    (void)do_update; // no preconditioner is used

    ReductionControl     reduction_control;
    SolverCG<VectorType> solver(reduction_control);
    solver.solve(op, dst, src, PreconditionIdentity());

    return reduction_control.last_step();
  }

  const Operator &op;
};

template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  void
  run()
  {
    using Operator     = NonlinearOperator<dim,
                                       fe_degree,
                                       n_points_1D,
                                       1,
                                       Number,
                                       VectorizedArrayType>;
    using LinearSolver = SolverCGWrapper<Operator>;

    Operator     nonlinear_operator;
    LinearSolver linear_solver(nonlinear_operator);

    NewtonSolver<VectorType, Operator, LinearSolver> newton_solver(
      nonlinear_operator, linear_solver);

    VectorType solution;

    newton_solver.solve(solution);
  }
};


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Test<2, 1>                       runner;
  runner.run();
}