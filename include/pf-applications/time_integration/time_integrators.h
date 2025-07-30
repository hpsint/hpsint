// ---------------------------------------------------------------------
//
// Copyright (C) 2023-2025 by the hpsint authors
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

#include <pf-applications/base/fe_integrator.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/time_integration/solution_history.h>
#include <pf-applications/time_integration/time_schemes.h>

#include <array>

namespace TimeIntegration
{
  using namespace dealii;

  template <typename Number>
  class TimeIntegratorData
  {
  public:
    // No implicit time scheme has been supplied
    TimeIntegratorData()
      : scheme(nullptr)
      , order(0)
      , weights{1}
      , stationary_weight(0)
    {}

    TimeIntegratorData(std::unique_ptr<ImplicitScheme<Number>> scheme_in)
      : scheme(std::move(scheme_in))
      , order(scheme ? scheme->get_order() : 0)
      , dt(order)
      , weights(order + 1)
      , stationary_weight(scheme ? 1 : 0)
    {}

    TimeIntegratorData(std::unique_ptr<ImplicitScheme<Number>> scheme_in,
                       Number                                  dt_init)
      : TimeIntegratorData(std::move(scheme_in))
    {
      update_dt(dt_init);
    }

    TimeIntegratorData(const TimeIntegratorData               &other,
                       std::unique_ptr<ImplicitScheme<Number>> scheme_in)
      : TimeIntegratorData(std::move(scheme_in))
    {
      for (unsigned int i = 0; i < std::min(order, other.order); ++i)
        dt[i] = other.dt[i];

      update_weights();
    }

    TimeIntegratorData &
    operator=(const TimeIntegratorData &other)
    {
      if (this != &other)
        {
          scheme = other.scheme ? other.scheme->clone() : nullptr;

          order             = other.order;
          dt                = other.dt;
          weights           = other.weights;
          stationary_weight = other.stationary_weight;
        }
      return *this;
    }

    TimeIntegratorData(const TimeIntegratorData &other)
    {
      *this = other;
    }

    void
    replace_dt(Number dt_new)
    {
      dt[0] = dt_new;

      update_weights();
    }

    void
    update_dt(Number dt_new)
    {
      for (int i = get_order() - 2; i >= 0; i--)
        {
          dt[i + 1] = dt[i];
        }

      dt[0] = dt_new;

      update_weights();
    }

    void
    copy_all_dt(const TimeIntegratorData<Number> &other)
    {
      AssertDimension(dt.size(), other.dt.size());
      dt = other.dt;

      update_weights();
    }

    const std::vector<Number> &
    get_all_dt() const
    {
      return dt;
    }

    Number
    get_current_dt() const
    {
      return dt[0];
    }

    // Weight for the current time step derivative
    Number
    get_primary_weight() const
    {
      return weights[0];
    }

    // Weight for the algebraic equations which can be used in the DAE
    // system being solved along with the time dependent PDEs
    Number
    get_algebraic_weight() const
    {
      return stationary_weight;
    }

    // All weights of the integration scheme
    const std::vector<Number> &
    get_weights() const
    {
      return weights;
    }

    unsigned int
    get_order() const
    {
      return order;
    }

    /* Serialization */
    template <class Archive>
    void
    save(Archive &ar, const unsigned int /*version*/) const
    {
      ar &order;
      ar &boost::serialization::make_array(dt.data(), dt.size());
    }

    template <class Archive>
    void
    load(Archive &ar, const unsigned int /*version*/)
    {
      ar &order;
      dt.resize(order);
      weights.resize(order + 1);

      ar &boost::serialization::make_array(dt.data(), dt.size());

      update_weights();
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    unsigned int
    effective_order() const
    {
      return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
        return v > 0;
      });
    }

  private:
    void
    update_weights()
    {
      if (scheme)
        weights = scheme->compute_weights(dt);
    }

    std::unique_ptr<ImplicitScheme<Number>> scheme;

    unsigned int        order;
    std::vector<Number> dt;
    std::vector<Number> weights;

    Number stationary_weight; // Weight for the algebraic equations
  };

  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  struct TimeCellIntegrator
  {
    using CellIntegrator =
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>;

    /* Here it is implied that weights.size() = order + 1 */
    template <typename Iterator>
    TimeCellIntegrator(const CellIntegrator &cell_integrator,
                       Iterator              begin,
                       Iterator              end)
      : order(std::distance(begin, end) - 1)
      , weights(begin, end)
      , evals(order, cell_integrator)
    {}

    unsigned int
    get_order() const
    {
      return order;
    }

    const std::vector<Number> &
    get_weights() const
    {
      return weights;
    }

    CellIntegrator &
    operator[](unsigned int i)
    {
      return evals[i];
    }

    const CellIntegrator &
    operator[](unsigned int i) const
    {
      return evals[i];
    }

  private:
    unsigned int                order;
    std::vector<Number>         weights;
    std::vector<CellIntegrator> evals;
  };

  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  void
  compute_time_derivative(
    VectorizedArrayType       &value_result,
    const VectorizedArrayType &val,
    const TimeCellIntegrator<dim, n_comp, Number, VectorizedArrayType>
                      &time_phi,
    const unsigned int index,
    const unsigned int q)
  {
    const auto &weights = time_phi.get_weights();

    value_result += val * weights[0];
    for (unsigned int i = 0; i < time_phi.get_order(); ++i)
      {
        const auto val_old = time_phi[i].get_value(q);

        value_result += val_old[index] * weights[i + 1];
      }
  }

  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  void
  compute_time_derivative(
    VectorizedArrayType &value_result,
    const FECellIntegratorValue<dim, n_comp, Number, VectorizedArrayType> &val,
    const TimeCellIntegrator<dim, n_comp, Number, VectorizedArrayType>
                      &time_phi,
    const unsigned int index,
    const unsigned int q)
  {
    compute_time_derivative(value_result, val[index], time_phi, index, q);
  }
} // namespace TimeIntegration