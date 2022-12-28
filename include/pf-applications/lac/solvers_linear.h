#pragma once

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_idr.h>

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
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "gmres::solve");

      typename SolverGMRES<T>::AdditionalData additional_data;

      if (data.orthogonalization_strategy == "classical gram schmidt")
        additional_data.orthogonalization_strategy = SolverGMRES<
          T>::AdditionalData::OrthogonalizationStrategy::classical_gram_schmidt;
      else if (data.orthogonalization_strategy == "modified gram schmidt")
        additional_data.orthogonalization_strategy = SolverGMRES<
          T>::AdditionalData::OrthogonalizationStrategy::modified_gram_schmidt;
      else
        AssertThrow(false, ExcNotImplemented());

      SolverGMRES<T> solver(solver_control, additional_data);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;
    SolverControl & solver_control;
    const GMRESData data;

    mutable MyTimerOutput timer;
  };



  template <typename Operator, typename Preconditioner>
  class SolverIDRWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverIDRWrapper(const Operator &op,
                     Preconditioner &preconditioner,
                     SolverControl & solver_control)
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "idr::solve");

      SolverIDR<T> solver(solver_control);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;
    SolverControl & solver_control;

    mutable MyTimerOutput timer;
  };



  template <typename Operator, typename Preconditioner>
  class SolverBicgstabWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverBicgstabWrapper(const Operator &   op,
                          Preconditioner &   preconditioner,
                          SolverControl &    solver_control,
                          const unsigned int max_bicgsteps)
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
      , max_bicgsteps(max_bicgsteps)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "bicgstab::solve");

      // copy information to be able to restart with GMRESs
      const unsigned int max_steps = solver_control.max_steps();

      T dst_copy;
      dst_copy.reinit(dst);
      dst_copy.copy_locally_owned_data_from(dst);

      try
        {
          // try to solve with Bicgstab with a reasonable number
          // of iterations
          solver_control.set_max_steps(max_bicgsteps);

          SolverBicgstab<T> solver(solver_control);
          solver.solve(op, dst, src, preconditioner);

          solver_control.set_max_steps(max_steps);

          return solver_control.last_step();
        }
      catch (const SolverControl::NoConvergence &)
        {
          // reset data
          solver_control.set_max_steps(max_steps);
          dst.copy_locally_owned_data_from(dst_copy);

          // solve with GMRES
          SolverGMRES<T> solver(solver_control);
          solver.solve(op, dst, src, preconditioner);

          return solver_control.last_step() + max_bicgsteps;
        }
    }

    const Operator &   op;
    Preconditioner &   preconditioner;
    SolverControl &    solver_control;
    const unsigned int max_bicgsteps;

    mutable MyTimerOutput timer;
  };

} // namespace LinearSolvers
