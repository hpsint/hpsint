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

} // namespace hpsint