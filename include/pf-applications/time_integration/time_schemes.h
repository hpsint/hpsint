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

  template <typename Number, typename Scheme>
  struct ImplicitSchemeBase : public ImplicitScheme<Number>
  {
    virtual std::unique_ptr<ImplicitScheme<Number>>
    clone() const override
    {
      return std::make_unique<Scheme>(static_cast<const Scheme &>(*this));
    }
  };

  template <typename Number>
  struct BDF1Scheme : public ImplicitSchemeBase<Number, BDF1Scheme<Number>>
  {
    unsigned int
    get_order() const override
    {
      return 1;
    }

    virtual std::vector<Number>
    compute_weights(const std::vector<Number> &dt) const override
    {
      std::vector<Number> weights(dt.size() + 1, 0.);
      if (this->effective_order(dt) == 1)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }

      return weights;
    }
  };

  template <typename Number>
  struct BDF2Scheme : public ImplicitSchemeBase<Number, BDF2Scheme<Number>>
  {
    unsigned int
    get_order() const override
    {
      return 2;
    }

    virtual std::vector<Number>
    compute_weights(const std::vector<Number> &dt) const override
    {
      std::vector<Number> weights(dt.size() + 1, 0.);
      if (this->effective_order(dt) == 2)
        {
          weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
          weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
          weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
        }
      else if (this->effective_order(dt) == 1)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }

      return weights;
    }
  };

  template <typename Number>
  struct BDF3Scheme : public ImplicitSchemeBase<Number, BDF3Scheme<Number>>
  {
    unsigned int
    get_order() const override
    {
      return 3;
    }

    virtual std::vector<Number>
    compute_weights(const std::vector<Number> &dt) const override
    {
      std::vector<Number> weights(dt.size() + 1, 0.);
      if (this->effective_order(dt) == 3)
        {
          weights[1] = -(dt[0] + dt[1]) * (dt[0] + dt[1] + dt[2]) /
                       (dt[0] * dt[1] * (dt[1] + dt[2]));
          weights[2] =
            dt[0] * (dt[0] + dt[1] + dt[2]) / (dt[1] * dt[2] * (dt[0] + dt[1]));
          weights[3] = -dt[0] * (dt[0] + dt[1]) /
                       (dt[2] * (dt[1] + dt[2]) * (dt[0] + dt[1] + dt[2]));
          weights[0] = -(weights[1] + weights[2] + weights[3]);
        }
      else if (this->effective_order(dt) == 2)
        {
          weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
          weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
          weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
        }
      else if (this->effective_order(dt) == 1)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }

      return weights;
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

  struct ForwardEulerScheme : public ExplicitScheme
  {
    std::vector<std::pair<double, double>>
    get_stages() const override
    {
      return {{1., 0.}};
    }
  };

  struct RungeKutta4Scheme : public ExplicitScheme
  {
    std::vector<std::pair<double, double>>
    get_stages() const override
    {
      return {{1. / 6., 1. / 2.},
              {1. / 3., 1. / 2.},
              {1. / 3., 1.},
              {1. / 6., 0.}};
    }
  };

  template <typename Number>
  struct IntegrationSchemeVariant
  {
    template <typename T>
    IntegrationSchemeVariant(std::unique_ptr<T> scheme)
      : scheme(std::move(scheme))
    {}

    template <typename T>
    std::unique_ptr<T>
    try_take()
    {
      if (std::holds_alternative<std::unique_ptr<T>>(scheme))
        return std::move(std::get<std::unique_ptr<T>>(scheme));
      else
        return nullptr;
    }

  private:
    std::variant<std::unique_ptr<ImplicitScheme<Number>>,
                 std::unique_ptr<ExplicitScheme>>
      scheme;
  };

  template <typename Number>
  IntegrationSchemeVariant<Number>
  create_time_scheme(std::string name)
  {
    static std::map<std::string,
                    std::function<IntegrationSchemeVariant<Number>()>>
      factory = {
        {"BDF1", []() { return std::make_unique<BDF1Scheme<Number>>(); }},
        {"BDF2", []() { return std::make_unique<BDF2Scheme<Number>>(); }},
        {"BDF3", []() { return std::make_unique<BDF3Scheme<Number>>(); }},
        {"FE", []() { return std::make_unique<ForwardEulerScheme>(); }},
        {"RK4", []() { return std::make_unique<RungeKutta4Scheme>(); }}};

    return factory.at(name)();
  }
} // namespace TimeIntegration