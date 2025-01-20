// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <deal.II/base/table_handler.h>

#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <pf-applications/base/mask.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/numerics/power_helper.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/time_integration/solution_history.h>

namespace Sintering
{
  using namespace dealii;

  template <typename VectorizedArrayType>
  class GreenquistFreeEnergy
  {
  private:
    using Number = double;

  public:
    static const unsigned int op_components_offset  = 2;
    static const bool         concentration_as_void = true;

    GreenquistFreeEnergy(double A, double B)
    {
      (void)A;
      (void)B;
    }

    // We still need Mask param since it is used in the preconditioners,
    // however, for this energy functional it is not used.
    template <typename Mask, int n_grains>
    Evaluation
    eval(const VectorizedArrayType *state) const
    {
      return Evaluation(state, std::integral_constant<int, n_grains>());
    }

    template <typename Mask>
    Evaluation
    eval(const VectorizedArrayType *state, const unsigned int n_grains) const
    {
      return Evaluation(state, n_grains);
    }

    template <typename Mask, typename VectorType>
    Evaluation
    eval(const VectorType &state, const unsigned int n_grains) const
    {
      return Evaluation(state, n_grains);
    }

    template <typename Mask, int n_grains, typename VectorType>
    Evaluation
    eval(const VectorType &state) const
    {
      return Evaluation(state, std::integral_constant<int, n_grains>());
    }

    template <typename Mask, typename VectorType>
    Evaluation
    eval(const VectorType &state) const
    {
      return Evaluation(state);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE void
    apply_d2f_detaidetaj(const VectorType &         L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType *      value_result) const
    {
      (void)L;
      (void)etas;
      (void)n_grains;
      (void)value;
      (void)value_result;
    }
  };
} // namespace Sintering
