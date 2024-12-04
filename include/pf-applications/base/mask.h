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

#include <type_traits>

namespace hpsint
{
  template <int N, int... M>
  struct Mask
  {
    static int const value = (1 << (N - 1)) | Mask<M...>::value;
  };

  template <int N>
  struct Mask<N>
  {
    static int const value = (1 << (N - 1));
  };

  template <>
  struct Mask<0>
  {
    static int const value = 0;
  };

  template <typename M, int bit>
  struct is_bit_set : std::integral_constant<
                        bool,
                        std::conditional_t<((M::value & (1 << bit - 1)) > 0),
                                           std::true_type,
                                           std::false_type>{}>
  {};

  template <typename M, int bit>
  constexpr bool is_bit_set_v = is_bit_set<M, bit>::value;

  template <typename M, int bit, int... bits>
  struct any_bit_of
    : std::integral_constant<bool,
                             std::conditional_t<is_bit_set_v<M, bit>,
                                                std::true_type,
                                                any_bit_of<M, bits...>>{}>
  {};

  template <typename M, int bit>
  struct any_bit_of<M, bit>
    : std::integral_constant<bool,
                             std::conditional_t<is_bit_set_v<M, bit>,
                                                std::true_type,
                                                std::false_type>{}>
  {};

  template <typename M, int bit, int... bits>
  constexpr bool any_bit_of_v = any_bit_of<M, bit, bits...>::value;

  template <typename E>
  constexpr typename std::underlying_type<E>::type
  to_underlying(E e) noexcept
  {
    return static_cast<typename std::underlying_type<E>::type>(e);
  }

  enum class EnergyEvaluation
  {
    zero   = 1,
    first  = 2,
    second = 3
  };

  using EnergyZero   = Mask<to_underlying(EnergyEvaluation::zero)>;
  using EnergyFirst  = Mask<to_underlying(EnergyEvaluation::first)>;
  using EnergySecond = Mask<to_underlying(EnergyEvaluation::second)>;
  using EnergyAll    = Mask<to_underlying(EnergyEvaluation::zero),
                         to_underlying(EnergyEvaluation::first),
                         to_underlying(EnergyEvaluation::second)>;

  template <typename M, EnergyEvaluation B, EnergyEvaluation... Bs>
  struct any_energy_eval_of
    : std::integral_constant<
        bool,
        std::conditional_t<is_bit_set_v<M, to_underlying(B)>,
                           std::true_type,
                           any_energy_eval_of<M, Bs...>>{}>
  {};

  template <typename M, EnergyEvaluation B>
  struct any_energy_eval_of<M, B>
    : std::integral_constant<
        bool,
        std::conditional_t<is_bit_set_v<M, to_underlying(B)>,
                           std::true_type,
                           std::false_type>{}>
  {};

  template <typename M, EnergyEvaluation B, EnergyEvaluation... Bs>
  constexpr bool any_energy_eval_of_v = any_energy_eval_of<M, B, Bs...>::value;
} // namespace hpsint