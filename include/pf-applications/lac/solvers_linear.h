#pragma once

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename Number>
  class LinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual ~LinearSolverBase() = default;

    virtual unsigned int
    solve(VectorType &dst, const VectorType &src) = 0;

    virtual unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) = 0;
  };



  template <typename Operator, typename Preconditioner>
  class SolverGMRESWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverGMRESWrapper(const Operator &op,
                       Preconditioner &preconditioner,
                       SolverControl & solver_control)
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      MyScope scope(timer, "gmres::solve");

      SolverGMRES<VectorType> solver(solver_control);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      MyScope scope(timer, "gmres::solve");

      SolverGMRES<BlockVectorType> solver(solver_control);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;
    SolverControl & solver_control;

    mutable MyTimerOutput timer;
  };
} // namespace LinearSolvers
