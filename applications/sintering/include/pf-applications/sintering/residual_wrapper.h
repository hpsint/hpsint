// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <pf-applications/lac/solvers_nonlinear.h>

#include <functional>

namespace Sintering
{
  using namespace dealii;
  using namespace NonLinearSolvers;

  template <typename BlockVectorType, typename OperatorType>
  class ResidualWrapper
  {
  public:
    using Callback = std::function<
      void(BlockVectorType &      dst,
           const BlockVectorType &src,
           const std::function<void(const unsigned int, const unsigned int)>,
           const std::function<void(const unsigned int, const unsigned int)>)>;

    ResidualWrapper(const OperatorType &op,
                    Callback            pre_callback  = {},
                    Callback            post_callback = {})
      : op(op)
      , pre_callback(pre_callback)
      , post_callback(post_callback)
    {}

    template <unsigned int with_time_derivative = 2>
    void
    evaluate_nonlinear_residual(
      BlockVectorType &      dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const
    {
      if (pre_callback)
        pre_callback(dst, src, pre_operation, post_operation);

      if constexpr (has_evaluate_nonlinear_residual_with_pre_post<
                      OperatorType,
                      BlockVectorType>)
        op.template evaluate_nonlinear_residual<with_time_derivative>(
          dst, src, pre_operation, post_operation);
      else
        op.template evaluate_nonlinear_residual<with_time_derivative>(dst, src);

      if (post_callback)
        post_callback(dst, src, pre_operation, post_operation);
    }

  private:
    const OperatorType &op;
    Callback            pre_callback;
    Callback            post_callback;
  };
} // namespace Sintering
