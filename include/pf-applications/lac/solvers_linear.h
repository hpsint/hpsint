#pragma once

#include <pf-applications/base/timer.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename Number>
  class LinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    virtual ~LinearSolverBase() = default;

    virtual unsigned int
    solve(VectorType &dst, const VectorType &src) = 0;
  };



  template <typename Operator, typename Preconditioner>
  class SolverGMRESWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType = typename Operator::vector_type;

    SolverGMRESWrapper(const Operator &op, Preconditioner &preconditioner)
      : op(op)
      , preconditioner(preconditioner)
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

      unsigned int            max_iter = 1000;
      ReductionControl        reduction_control(max_iter, 1.e-10, 1.e-2);
      SolverGMRES<VectorType> solver(reduction_control);
      solver.solve(op, dst, src, preconditioner);

      return reduction_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
  };
} // namespace LinearSolvers
