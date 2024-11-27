// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the hpsint authors
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

#include <pf-applications/sintering/initial_values_array.h>

namespace Sintering
{
  template <int dim>
  class InitialValuesHypercube : public InitialValuesArray<dim>
  {
  public:
    InitialValuesHypercube(
      const double                         r0,
      const double                         interface_width,
      const std::array<unsigned int, dim> &n_grains,
      const unsigned int                   n_order_parameters,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset = 2,
      const bool               is_accumulative      = false)
      : InitialValuesArray<dim>(r0,
                                interface_width,
                                interface_direction,
                                op_components_offset,
                                is_accumulative)
    {
      unsigned int counter = 0;

      // If n_order_parameters == 0, then do not minimize
      if (n_order_parameters > 0)
        {
          const unsigned int n_total_grains =
            std::accumulate(n_grains.begin(),
                            n_grains.end(),
                            1,
                            std::multiplies<unsigned int>());

          for (unsigned int op = 0;
               op < std::min(n_total_grains, n_order_parameters);
               ++op)
            this->order_parameter_to_grains[op];
        }

      if (dim == 2)
        {
          for (unsigned int i = 0; i < n_grains[0]; ++i)
            for (unsigned int j = 0; j < n_grains[1]; ++j)
              {
                this->centers.emplace_back(2 * r0 * i, 2 * r0 * j);
                assign_order_parameter(i + j, counter++, n_order_parameters);
              }
        }
      else if (dim == 3)
        {
          for (unsigned int i = 0; i < n_grains[0]; ++i)
            for (unsigned int j = 0; j < n_grains[1]; ++j)
              for (unsigned int k = 0; k < n_grains[2]; ++k)
                {
                  this->centers.emplace_back(2 * r0 * i,
                                             2 * r0 * j,
                                             2 * r0 * k);
                  assign_order_parameter(i + j + k,
                                         counter++,
                                         n_order_parameters);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  private:
    void
    assign_order_parameter(const unsigned int order,
                           const unsigned int counter,
                           const unsigned int n_order_parameters)
    {
      if (n_order_parameters > 0)
        this->order_parameter_to_grains[order % n_order_parameters].push_back(
          counter);
      else
        this->order_parameter_to_grains[counter] = {counter};
    }
  };
} // namespace Sintering