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

#include <array>
#include <memory>
#include <utility>

namespace internal
{
  template <typename T, std::size_t... Is>
  constexpr std::array<T, sizeof...(Is)>
  create_array(T value, std::index_sequence<Is...>)
  {
    /* Here the following initializer list is generated:
     * {(void, value), (void, value), ..., (void, value)}
     * containing N expressions (void, value) such that each i-th element of the
     * array is assigned to value. Such initialization uses the property of the
     * comma operator to join together multiple expressions. The result of the
     * assignement is then the value of the right-most expression, e.g.:
     * int i = (1, 2, 3); // i = 3
     *
     * static_cast is used here to remove the warning: unused value */
    return {{(static_cast<void>(Is), value)...}};
  }
} // namespace internal

/* This function creates a std::array of a given size N with all of its elements
 * initialized with the given value. Distinctly from std::array, first the
 * template parameter is the array length whereas the second in the data type.
 * This is done in order to have a possiblity to avoid specitying T at all such
 * that it will be deduced automatically. */
template <std::size_t N, typename T>
constexpr std::array<T, N>
create_array(const T &value)
{
  /* Here std::make_index_sequence<N>() creates an sequence of std::size_t -
   * namely std::integer_sequence<std::size_t, 0, 1, ..., N> - via the variadic
   * templates mechanism and this is then send to another function. */
  return ::internal::create_array(value, std::make_index_sequence<N>());
}

template <typename T>
std::unique_ptr<std::decay_t<T>>
make_unique_from(T &&obj)
{
  using U = std::decay_t<T>;
  return std::make_unique<U>(std::forward<T>(obj));
}
