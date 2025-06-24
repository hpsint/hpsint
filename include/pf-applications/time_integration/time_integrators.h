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

#include <pf-applications/base/fe_integrator.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/time_integration/solution_history.h>

#include <array>

namespace TimeIntegration
{
  using namespace dealii;

  unsigned int
  get_scheme_order(std::string scheme)
  {
    unsigned int time_integration_order = 0;
    if (scheme == "BDF1")
      time_integration_order = 1;
    else if (scheme == "BDF2")
      time_integration_order = 2;
    else if (scheme == "BDF3")
      time_integration_order = 3;
    else
      AssertThrow(false, ExcNotImplemented());

    return time_integration_order;
  }

  template <typename Number>
  class TimeIntegratorData
  {
  public:
    TimeIntegratorData(unsigned int order)
      : order(order)
      , dt(order)
      , weights(order + 1)
    {}

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

  private:
    unsigned int
    effective_order() const
    {
      return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
        return v > 0;
      });
    }

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

  template <int dim, typename Number, typename VectorizedArrayType>
  class BDFIntegrator
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    template <int n_comp>
    using CellIntegrator =
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>;

    template <int n_comp>
    using CellIntegratorValue =
      typename FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>::
        value_type;

    template <int n_comp>
    using TimeCellIntegrator =
      std::vector<FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>>;

    BDFIntegrator(const TimeIntegratorData<Number>       &time_data,
                  const SolutionHistory<BlockVectorType> &history)
      : time_data(time_data)
      , history(history)
    {}

    template <int n_comp>
    void
    compute_time_derivative(VectorizedArrayType               &value_result,
                            const CellIntegratorValue<n_comp> &val,
                            const TimeCellIntegrator<n_comp>  &time_phi,
                            const unsigned int                 index,
                            const unsigned int                 q) const
    {
      compute_time_derivative(value_result, val[index], time_phi, index, q);
    }

    template <int n_comp>
    void
    compute_time_derivative(VectorizedArrayType              &value_result,
                            const VectorizedArrayType        &val,
                            const TimeCellIntegrator<n_comp> &time_phi,
                            const unsigned int                index,
                            const unsigned int                q) const
    {
      AssertThrow(time_data.get_order() == time_phi.size(),
                  ExcMessage("Inconsistent data structures provided!"));

      const auto &weights = time_data.get_weights();

      value_result += val * weights[0];
      for (unsigned int i = 0; i < time_data.get_order(); ++i)
        {
          const auto val_old = time_phi[i].get_value(q);

          value_result += val_old[index] * weights[i + 1];
        }
    }

    template <int n_comp>
    TimeCellIntegrator<n_comp>
    create_cell_intergator(const CellIntegrator<n_comp> &cell_integrator) const
    {
      return TimeCellIntegrator<n_comp>(time_data.get_order(), cell_integrator);
    }

  private:
    const TimeIntegratorData<Number>       &time_data;
    const SolutionHistory<BlockVectorType> &history;
  };
} // namespace TimeIntegration