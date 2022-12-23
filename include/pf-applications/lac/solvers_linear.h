#pragma once

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/solvers_linear_parameters.h>

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

    SolverGMRESWrapper(const Operator & op,
                       Preconditioner & preconditioner,
                       SolverControl &  solver_control,
                       const GMRESData &data = GMRESData())
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
      , data(data)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      MyScope scope(timer, "gmres::solve");

      typename SolverGMRES<VectorType>::AdditionalData additional_data;

      if (data.orthogonalization_strategy == "classical gram schmidt")
        additional_data.orthogonalization_strategy = SolverGMRES<VectorType>::
          AdditionalData::OrthogonalizationStrategy::classical_gram_schmidt;
      else if (data.orthogonalization_strategy == "modified gram schmidt")
        additional_data.orthogonalization_strategy = SolverGMRES<VectorType>::
          AdditionalData::OrthogonalizationStrategy::modified_gram_schmidt;
      else
        AssertThrow(false, ExcNotImplemented());

      SolverGMRES<VectorType> solver(solver_control, additional_data);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      MyScope scope(timer, "gmres::solve");

      typename SolverGMRES<BlockVectorType>::AdditionalData additional_data;

      if (data.orthogonalization_strategy == "classical gram schmidt")
        additional_data.orthogonalization_strategy =
          SolverGMRES<BlockVectorType>::AdditionalData::
            OrthogonalizationStrategy::classical_gram_schmidt;
      else if (data.orthogonalization_strategy == "modified gram schmidt")
        additional_data.orthogonalization_strategy =
          SolverGMRES<BlockVectorType>::AdditionalData::
            OrthogonalizationStrategy::modified_gram_schmidt;
      else
        AssertThrow(false, ExcNotImplemented());

      SolverGMRES<BlockVectorType> solver(solver_control, additional_data);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

  private:
    const Operator &op;
    Preconditioner &preconditioner;
    SolverControl & solver_control;
    const GMRESData data;

    mutable MyTimerOutput timer;
  };
} // namespace LinearSolvers
