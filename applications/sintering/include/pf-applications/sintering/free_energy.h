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

#include <pf-applications/base/mask.h>

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

    // This class knows about the structure of the state vector
    template <bool with_power_3>
    class Evaluation
    {
    public:
      template <int n_grains>
      Evaluation(const double               A,
                 const double               B,
                 const VectorizedArrayType *state,
                 std::integral_constant<int, n_grains>)
        : A(A)
        , B(B)
        , c(state[0])
      {
        const VectorizedArrayType *etas = &state[0] + 2;

        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);
        if constexpr (with_power_3)
          etaPower3Sum = PowerHelper<n_grains, 3>::power_sum(etas);
      }

      template <typename VectorType, int n_grains>
      Evaluation(const double      A,
                 const double      B,
                 const VectorType &state,
                 std::integral_constant<int, n_grains>)
        : A(A)
        , B(B)
        , c(state[0])
      {
        std::array<VectorizedArrayType, n_grains> etas;
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          etas[ig] = state[2 + ig];

        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);
        if constexpr (with_power_3)
          etaPower3Sum = PowerHelper<n_grains, 3>::power_sum(etas);
      }

      template <typename VectorType>
      Evaluation(const double       A,
                 const double       B,
                 const VectorType & state,
                 const unsigned int n_grains)
        : A(A)
        , B(B)
        , c(state[0])
      {
        std::vector<VectorizedArrayType> etas(n_grains);
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          etas[ig] = state[2 + ig];

        etaPower2Sum = PowerHelper<0, 2>::power_sum(etas);
        if constexpr (with_power_3)
          etaPower3Sum = PowerHelper<0, 3>::power_sum(etas);
      }

      template <typename VectorType,
                int n_components = SizeHelper<VectorType>::size>
      Evaluation(const double A, const double B, const VectorType &state)
        : A(A)
        , B(B)
        , c(state[0])
      {
        constexpr int n_grains = n_components - 2;

        std::array<VectorizedArrayType, n_grains> etas;
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          etas[ig] = state[2 + ig];

        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);
        if constexpr (with_power_3)
          etaPower3Sum = PowerHelper<n_grains, 3>::power_sum(etas);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      f() const
      {
        Assert(with_power_3,
               ExcMessage("The evaluator was initialized without "
                          " parameter with_power_3 enabled"));

        return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
               B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                    (-4.0 * c + 8.0) * etaPower3Sum +
                    3.0 * (etaPower2Sum * etaPower2Sum));
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      df_dc() const
      {
        Assert(with_power_3,
               ExcMessage("The evaluator was initialized without "
                          " parameter with_power_3 enabled"));

        return A * (c * c) * (2.0 * c - 2.0) +
               2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
               B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      df_detai(const VectorizedArrayType &etai) const
      {
        return etai * (B * 12.0) *
               (etai * (1.0 * c - 2.0) + (-c + 1.0) + etaPower2Sum);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_dc2() const
      {
        return 2.0 * A * (c * c) + 4.0 * A * c * (2.0 * c - 2.0) +
               2.0 * A * ((-c + 1.0) * (-c + 1.0)) + 2.0 * B;
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_dcdetai(const VectorizedArrayType &etai) const
      {
        return (B * 12.0) * etai * (etai - 1.0);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detai2(const VectorizedArrayType &etai) const
      {
        return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                    24.0 * (etai * etai) + 12.0 * etaPower2Sum);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detaidetaj(const VectorizedArrayType &etai,
                     const VectorizedArrayType &etaj) const
      {
        return 24.0 * B * etai * etaj;
      }

    private:
      double              A;
      double              B;
      VectorizedArrayType c;
      VectorizedArrayType etaPower2Sum;
      VectorizedArrayType etaPower3Sum;
    };

    template <typename Mask>
    using EvaluationConcrete =
      Evaluation<any_energy_eval_of_v<Mask,
                                      EnergyEvaluation::zero,
                                      EnergyEvaluation::first>>;

  public:
    FreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <typename Mask, int n_grains>
    EvaluationConcrete<Mask>
    eval(const VectorizedArrayType *state) const
    {
      return EvaluationConcrete<Mask>(A,
                                      B,
                                      state,
                                      std::integral_constant<int, n_grains>());
    }

    template <typename Mask>
    EvaluationConcrete<Mask>
    eval(const VectorizedArrayType *state, const unsigned int n_grains) const
    {
      return EvaluationConcrete<Mask>(A, B, state, n_grains);
    }

    template <typename Mask, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state, const unsigned int n_grains) const
    {
      return EvaluationConcrete<Mask>(A, B, state, n_grains);
    }

    template <typename Mask, int n_grains, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state) const
    {
      return EvaluationConcrete<Mask>(A,
                                      B,
                                      state,
                                      std::integral_constant<int, n_grains>());
    }

    template <typename Mask, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state) const
    {
      return EvaluationConcrete<Mask>(A, B, state);
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
