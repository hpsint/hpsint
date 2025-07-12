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

#include <array>

namespace TimeIntegration
{
  using namespace dealii;

  unsigned int
  get_scheme_order(std::string scheme);

  template <typename Number>
  class TimeIntegratorData
  {
  public:
    TimeIntegratorData()
      : order(0)
    {}

    TimeIntegratorData(unsigned int order)
      : order(order)
      , dt(order)
      , weights(order + 1)
    {}

    TimeIntegratorData(unsigned int order, Number dt_init)
      : TimeIntegratorData(order)
    {
      update_dt(dt_init);
    }

    TimeIntegratorData(const TimeIntegratorData &other, unsigned int order)
      : TimeIntegratorData(order)
    {
      for (unsigned int i = 0; i < std::min(order, other.order); ++i)
        dt[i] = other.dt[i];

      update_weights();
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
    set_all_dt(const std::vector<Number> &dt_new)
    {
      AssertDimension(dt.size(), dt_new.size());
      dt = dt_new;

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

    // Weight for the algebraic equations which can be used in the DAE system
    // being solved along with the time dependent PDEs
    Number
    get_algebraic_weight() const
    {
      return 1.0;
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
      std::fill(weights.begin(), weights.end(), 0);

      if (effective_order() == 3)
        {
          weights[1] = -(dt[0] + dt[1]) * (dt[0] + dt[1] + dt[2]) /
                       (dt[0] * dt[1] * (dt[1] + dt[2]));
          weights[2] =
            dt[0] * (dt[0] + dt[1] + dt[2]) / (dt[1] * dt[2] * (dt[0] + dt[1]));
          weights[3] = -dt[0] * (dt[0] + dt[1]) /
                       (dt[2] * (dt[1] + dt[2]) * (dt[0] + dt[1] + dt[2]));
          weights[0] = -(weights[1] + weights[2] + weights[3]);
        }
      else if (effective_order() == 2)
        {
          weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
          weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
          weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
        }
      else if (effective_order() == 1)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }
      else
        {
          AssertThrow(effective_order() <= 3, ExcMessage("Not implemented"));
        }
    }

    unsigned int        order;
    std::vector<Number> dt;
    std::vector<Number> weights;
  };

  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  struct TimeCellIntegrator
  {
    using CellIntegrator =
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>;

    template <typename Iterator>
    TimeCellIntegrator(const CellIntegrator &cell_integrator,
                       Iterator              begin,
                       Iterator              end)
      : weights(begin, end)
      , evals(std::distance(begin, end), cell_integrator)
    {}

    unsigned int
    get_order() const
    {
      return weights.size();
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