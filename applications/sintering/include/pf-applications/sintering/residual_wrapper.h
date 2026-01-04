// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2025 by the hpsint authors
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

#include <pf-applications/lac/residual_wrapper.h>

#include <pf-applications/sintering/operator_advection.h>

#include <functional>

namespace Sintering
{
  using namespace dealii;
  using namespace NonLinearSolvers;

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

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename OperatorType,
            bool with_time_derivative = true>
  class ResidualWrapperAdvection
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

    ResidualWrapperAdvection(
      const AdvectionOperator<dim, Number, VectorizedArrayType>
                         &advection_operator,
      const OperatorType &nonlinear_operator)
      : ResidualWrapperBase<Number, OperatorType>(nonlinear_operator)
      , advection_operator(advection_operator)
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
      advection_operator.evaluate_forces(src, pre_operation, post_operation);
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
      advection_operator.evaluate_forces(src, pre_operation, post_operation);
      this->template do_evaluate_nonlinear_residual<with_time_derivative>(
        dst, src, pre_operation, post_operation);
    }

  private:
    const AdvectionOperator<dim, Number, VectorizedArrayType>
      &advection_operator;
  };
} // namespace Sintering
