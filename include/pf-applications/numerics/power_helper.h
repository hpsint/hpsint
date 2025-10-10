// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <pf-applications/base/tensor.h>

#include <array>
#include <numeric>
#include <vector>

namespace hpsint
{
  using namespace dealii;

  template <unsigned int n, std::size_t p>
  class PowerHelper
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      T initial = 0.0;

      for (unsigned int i = 0; i < n; ++i)
        initial += Utilities::fixed_power<p>(etas[i]);

      return initial;
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, n> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const Tensor<1, n, T> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        hpsint::cbegin(etas), hpsint::cend(etas), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::vector<T> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }
  };

  template <>
  class PowerHelper<2, 2>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const Tensor<1, 2, T> &etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }
  };

  template <>
  class PowerHelper<2, 3>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const Tensor<1, 2, T> &etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }
  };


  template <class T>
  class SizeHelper;

  template <class T, std::size_t n>
  class SizeHelper<std::array<T, n>>
  {
  public:
    static const std::size_t size = n;
  };

  template <int dim, class T>
  class SizeHelper<Tensor<1, dim, T>>
  {
  public:
    static const std::size_t size = dim;
  };

  template <class T>
  class SizeHelper<std::vector<T>>
  {
  public:
    static const std::size_t size = 0;
  };
} // namespace hpsint