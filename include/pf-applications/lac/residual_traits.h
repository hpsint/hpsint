// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

#include <deal.II/base/template_constraints.h>

#include <functional>
#include <utility>

namespace NonLinearSolvers
{
  template <typename T, typename VectorType>
  using evaluate_nonlinear_residual_with_pre_post_t =
    decltype(std::declval<const T>().evaluate_nonlinear_residual(
      std::declval<VectorType &>(),
      std::declval<const VectorType &>(),
      std::declval<
        const std::function<void(const unsigned int, const unsigned int)>>(),
      std::declval<
        const std::function<void(const unsigned int, const unsigned int)>>()));

  template <typename T, typename VectorType>
  constexpr bool has_evaluate_nonlinear_residual_with_pre_post =
    dealii::internal::is_supported_operation<
      evaluate_nonlinear_residual_with_pre_post_t,
      T,
      VectorType>;
} // namespace NonLinearSolvers