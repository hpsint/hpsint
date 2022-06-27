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
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {}

    ~SolverGMRESWrapper()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

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

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
  };
} // namespace LinearSolvers
