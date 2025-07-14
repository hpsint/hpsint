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

#include <pf-applications/base/data.h>

#include <functional>
#include <utility>
#include <variant>
#include <vector>

namespace TimeIntegration
{
  template <typename Number>
  struct ImplicitScheme
  {
    virtual ~ImplicitScheme() = default;

    // Returns order of the scheme
    virtual unsigned int
    get_order() const = 0;

    // Computes the weights for the scheme based on the time steps
    // dt.size() must be equal to get_order()
    virtual std::vector<Number>
    compute_weights(const std::vector<Number> &dt) const = 0;

    virtual std::unique_ptr<ImplicitScheme<Number>>
    clone() const = 0;

  protected:
    unsigned int
    effective_order(const std::vector<Number> &dt) const
    {
      return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
        return v > 0;
      });
    }
  };

  struct ExplicitScheme
  {
    virtual ~ExplicitScheme() = default;

    /* Returns stages of the explicit scheme. For each stage a pair is defined.
     * The first item p1_i says how the RHS vector f_i should be scaled when
     * adding it to the state vector increment, i.e. dy += p1_i * h * f_i, and
     * the second item p2_i says how f_i should be scaled when appending it to
     * y_n before computing f_(i+1). Ultimately, for the last stage, p2 = 0. */
    virtual std::vector<std::pair<double, double>>
    get_stages() const = 0;
  };
} // namespace TimeIntegration