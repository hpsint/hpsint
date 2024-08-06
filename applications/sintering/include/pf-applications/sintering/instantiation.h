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

#include <deal.II/base/exceptions.h>
#include <deal.II/base/template_constraints.h>

#include <boost/preprocessor.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <utility>

template <typename T>
using n_grains_t = decltype(std::declval<T const>().n_grains());

template <typename T>
using n_grains_to_n_components_t =
  decltype(std::declval<T const>().n_grains_to_n_components(
    std::declval<const unsigned int>()));

template <typename T>
constexpr bool has_n_grains_method =
  dealii::internal::is_supported_operation<n_grains_t, T>
    &&dealii::internal::is_supported_operation<n_grains_to_n_components_t, T>;

DeclException4(ExcInvalidNumberOfComponents,
               unsigned int,
               unsigned int,
               unsigned int,
               std::string,
               << "This operation is precompiled for the number of " << arg4
               << " in range [" << arg1 << ", " << arg2 << "] "
               << "but you provided n_" << arg4 << " = " << arg3);

// clang-format off
/**
 * Macro that converts a runtime number (n_components() or n_grains())
 * to constant expressions that can be used for templating and calles
 * the provided function with the two parameters: 1) number of
 * components and 2) number of grains (if it makes sence; else -1).
 *
 * The relation between number of components and number of grains
 * is encrypted in the method T::n_grains_to_n_components().
 * 
 * The function can be used the following way:
 * ```
 * #define OPERATION(c, d) std::cout << c << " " << d << std::endl;
 * EXPAND_OPERATIONS(OPERATION);
 * #undef OPERATION
 * ```
 */

#define EXPAND_CONST(z, n, data) \
  case n: { data(T::n_grains_to_n_components(std::min(max_grains, n)), std::min(max_grains, n)); break; }

#define EXPAND_NONCONST(z, n, data) \
  case n: { data(std::min(max_components, n), -1); break; }

#define EXPAND_MAX_SINTERING_COMPONENTS \
  BOOST_PP_ADD(MAX_SINTERING_GRAINS, 2)

#define EXPAND_OPERATIONS(OPERATION)                                                                                  \
  if constexpr(has_n_grains_method<T>)                                                                                \
    {                                                                                                                 \
      constexpr int max_grains = MAX_SINTERING_GRAINS;                                                                \
      const unsigned int n_grains = static_cast<const T&>(*this).n_grains();                                          \
      switch (n_grains)                                                                                               \
        {                                                                                                             \
          BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(MAX_SINTERING_GRAINS), EXPAND_CONST, OPERATION);                    \
          default:                                                                                                    \
            AssertThrow(false, ExcInvalidNumberOfComponents(1, MAX_SINTERING_GRAINS, n_grains, "grains"));            \
        }                                                                                                             \
    }                                                                                                                 \
  else                                                                                                                \
    {                                                                                                                 \
      constexpr int max_components = MAX_SINTERING_GRAINS + 2;                                                        \
      switch (this->n_components())                                                                                   \
        {                                                                                                             \
          BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(EXPAND_MAX_SINTERING_COMPONENTS), EXPAND_NONCONST, OPERATION);      \
          default:                                                                                                    \
            AssertThrow(false, ExcInvalidNumberOfComponents(1, max_components, this->n_components(), "components"));  \
        }                                                                                                             \
    }

#define EXPAND_OPERATIONS_N_COMP_NT(OPERATION)                                                               \
  constexpr int max_components = MAX_SINTERING_GRAINS + 2;                                                   \
  switch (n_comp_nt)                                                                                         \
    {                                                                                                        \
      BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(EXPAND_MAX_SINTERING_COMPONENTS), EXPAND_NONCONST, OPERATION); \
      default:                                                                                               \
        AssertThrow(false, ExcInvalidNumberOfComponents(2, max_components, n_comp_nt, "components"));        \
    }
// clang-format on