// ---------------------------------------------------------------------
//
// Copyright (C) 2023-2026 by the hpsint authors
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

#include <vector>

namespace TimeIntegration
{
  using namespace dealii;

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