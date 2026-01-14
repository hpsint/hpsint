// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#pragma once

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/residual_traits.h>

#include <functional>

namespace NonLinearSolvers
{
  using namespace dealii;

  template <typename Number>
  class ResidualWrapper : public Subscriptor
  {
  public:
    using value_type  = Number;
    using vector_type = LinearAlgebra::distributed::Vector<Number>;
    using VectorType  = vector_type;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    // This mimic the operator's interface
    template <int with_time_derivative = 2>
    void
    evaluate_nonlinear_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const
    {
      static_assert(with_time_derivative == 1 || with_time_derivative == 2);

      if constexpr (with_time_derivative == 2)
        evaluate_for_residual(dst, src, pre_operation, post_operation);
      else if (with_time_derivative == 1)
        evaluate_for_jacobian(dst, src, pre_operation, post_operation);
    }

  private:
    virtual void
    evaluate_for_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const = 0;

    virtual void
    evaluate_for_jacobian(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const = 0;
  };

  template <typename Number,
            typename OperatorType,
            bool with_time_derivative = true>
  class ResidualWrapperDirect : public ResidualWrapper<Number>
  {
  public:
    using value_type      = typename ResidualWrapper<Number>::value_type;
    using vector_type     = typename ResidualWrapper<Number>::vector_type;
    using VectorType      = typename ResidualWrapper<Number>::VectorType;
    using BlockVectorType = typename ResidualWrapper<Number>::BlockVectorType;

    ResidualWrapperDirect(const OperatorType &nonlinear_operator)
      : nonlinear_operator(nonlinear_operator)
    {}

  private:
    const OperatorType &nonlinear_operator;

    void
    evaluate_for_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      nonlinear_operator
        .template evaluate_nonlinear_residual<with_time_derivative ? 2 : 0>(
          dst, src, pre_operation, post_operation);
    }

    virtual void
    evaluate_for_jacobian(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      nonlinear_operator
        .template evaluate_nonlinear_residual<with_time_derivative>(
          dst, src, pre_operation, post_operation);
    }
  };

  template <typename Number, typename OperatorType>
  class ResidualWrapperBase : public ResidualWrapper<Number>
  {
  public:
    using value_type      = typename ResidualWrapper<Number>::value_type;
    using vector_type     = typename ResidualWrapper<Number>::vector_type;
    using VectorType      = typename ResidualWrapper<Number>::VectorType;
    using BlockVectorType = typename ResidualWrapper<Number>::BlockVectorType;

    ResidualWrapperBase(const OperatorType &nonlinear_operator)
      : nonlinear_operator(nonlinear_operator)
    {}

  protected:
    template <unsigned int with_time_derivative = 2>
    void
    do_evaluate_nonlinear_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const
    {
      if constexpr (has_evaluate_nonlinear_residual_with_pre_post<
                      OperatorType,
                      BlockVectorType>)
        {
          nonlinear_operator
            .template evaluate_nonlinear_residual<with_time_derivative>(
              dst, src, pre_operation, post_operation);
        }
      else
        {
          std::pair<unsigned int, unsigned int> range{
            0, nonlinear_operator.get_matrix_free().n_cell_batches()};

          if (pre_operation)
            pre_operation(range.first, range.second);

          nonlinear_operator
            .template evaluate_nonlinear_residual<with_time_derivative>(dst,
                                                                        src);

          if (post_operation)
            post_operation(range.first, range.second);
        }
    }

  private:
    const OperatorType &nonlinear_operator;
  };

  template <typename Number,
            typename OperatorType,
            bool with_time_derivative = true>
  class ResidualWrapperGeneric
    : public ResidualWrapperBase<Number, OperatorType>
  {
  public:
    using value_type =
      typename ResidualWrapperBase<Number, OperatorType>::value_type;
    using vector_type =
      typename ResidualWrapperBase<Number, OperatorType>::vector_type;
    using VectorType =
      typename ResidualWrapperBase<Number, OperatorType>::VectorType;
    using BlockVectorType =
      typename ResidualWrapperBase<Number, OperatorType>::BlockVectorType;

    ResidualWrapperGeneric(const OperatorType &nonlinear_operator)
      : ResidualWrapperBase<Number, OperatorType>(nonlinear_operator)
    {}

  private:
    void
    evaluate_for_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      this
        ->template do_evaluate_nonlinear_residual<with_time_derivative ? 2 : 0>(
          dst, src, pre_operation, post_operation);
    }

    void
    evaluate_for_jacobian(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      this->template do_evaluate_nonlinear_residual<with_time_derivative>(
        dst, src, pre_operation, post_operation);
    }
  };
} // namespace NonLinearSolvers
