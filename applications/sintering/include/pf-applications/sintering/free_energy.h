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

#include <deal.II/base/utilities.h>

#include <pf-applications/numerics/power_helper.h>

namespace Sintering
{
  using namespace dealii;
  using namespace hpsint;

  template <typename VectorizedArrayType>
  class FreeEnergy
  {
  private:
    double A;
    double B;

  public:
    FreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c,
      const VectorType &         etas,
      const VectorizedArrayType &etaPower2Sum,
      const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
             B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                  (-4.0 * c + 8.0) * etaPower3Sum +
                  3.0 * (etaPower2Sum * etaPower2Sum));
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return f(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c,
          const VectorType &         etas,
          const VectorizedArrayType &etaPower2Sum,
          const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * (2.0 * c - 2.0) +
             2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
             B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return df_dc(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             const VectorizedArrayType &etaPower2Sum,
             unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return etai * (B * 12.0) *
             (etai * (1.0 * c - 2.0) + (-c + 1.0) + etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return df_detai(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dc2(const VectorizedArrayType &c, const VectorType &etas) const
    {
      (void)etas;

      return 2.0 * A * (c * c) + 4.0 * A * c * (2.0 * c - 2.0) +
             2.0 * A * ((-c + 1.0) * (-c + 1.0)) + 2.0 * B;
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dcdetai(const VectorizedArrayType &c,
                const VectorType &         etas,
                unsigned int               index_i) const
    {
      (void)c;

      const auto &etai = etas[index_i];

      return (B * 12.0) * etai * (etai - 1.0);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               const VectorizedArrayType &etaPower2Sum,
               unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                  24.0 * (etai * etai) + 12.0 * etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return d2f_detai2(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detaidetaj(const VectorizedArrayType &c,
                   const VectorType &         etas,
                   unsigned int               index_i,
                   unsigned int               index_j) const
    {
      (void)c;

      const auto &etai = etas[index_i];
      const auto &etaj = etas[index_j];

      return 24.0 * B * etai * etaj;
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE void
    apply_d2f_detaidetaj(const VectorType &         L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType *      value_result) const
    {
      if (n_grains <= 1)
        return; // nothing to do

      VectorizedArrayType temp = etas[0] * value[0];

      for (unsigned int ig = 1; ig < n_grains; ++ig)
        temp += etas[ig] * value[ig];

      for (unsigned int ig = 0; ig < n_grains; ++ig)
        value_result[ig] +=
          (L * 24.0 * B) * etas[ig] * (temp - etas[ig] * value[ig]);
    }
  };

} // namespace Sintering
