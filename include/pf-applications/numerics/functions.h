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

template <typename Number>
class Function1D
{
public:
  virtual Number
  value(const Number x) const = 0;
};

template <typename Number>
class Function1DConstant : public Function1D<Number>
{
public:
  Function1DConstant(const double y)
    : y(y)
  {}

  Number
  value(const Number x) const override
  {
    return y;
  }

private:
  double y;
};

template <typename Number>
class Function1DPiecewise : public Function1D<Number>
{
public:
  Function1DPiecewise(const bool extrapolate_linear = false)
    : extrapolate_linear(extrapolate_linear)
  {}

  Function1DPiecewise(const std::map<Number, Number> &map_pairs,
                      const bool extrapolate_linear = false)
    : pairs(map_pairs.begin(), map_pairs.end())
    , extrapolate_linear(extrapolate_linear)
  {}

  void
  add_pair(const Number x, const Number y)
  {
    const auto location = std::upper_bound(
      pairs.begin(), pairs.end(), x, [](const auto &value, const auto &right) {
        return value < right.first;
      });
    pairs.emplace(location, x, y);
  }

  Number
  value(const Number x) const override
  {
    if (pairs.empty())
      {
        return 0.;
      }
    else
      {
        auto it = std::lower_bound(pairs.begin(),
                                   pairs.end(),
                                   x,
                                   [](const auto &left, const auto &value) {
                                     return left.first < value;
                                   });

        decltype(it) i1, i2;
        if (it == pairs.begin())
          {
            if (extrapolate_linear)
              {
                i1 = pairs.begin();
                i2 = pairs.begin();
                std::advance(i2, 1);
              }
            else
              {
                return it->second;
              }
          }
        else if (it == pairs.end())
          {
            if (extrapolate_linear)
              {
                i1 = pairs.end();
                std::advance(i1, -2);
                i2 = pairs.end();
                std::advance(i2, -1);
              }
            else
              {
                std::advance(it, -1);
                return it->second;
              }
          }
        else
          {
            i1 = it;
            std::advance(i1, -1);
            i2 = it;
          }

        const double k = (i1->second - i2->second) / (i1->first - i2->first);
        const double b = i1->second - k * i1->first;
        const double y = k * x + b;

        return y;
      }
  }

private:
  std::vector<std::pair<Number, Number>> pairs;

  const bool extrapolate_linear;
};
