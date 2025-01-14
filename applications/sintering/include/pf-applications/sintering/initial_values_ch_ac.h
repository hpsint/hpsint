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

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <pf-applications/sintering/initial_values.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim>
  class InitialValuesCHAC : public InitialValues<dim>
  {
  public:
    InitialValuesCHAC(
      const double             interface_width,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset  = 2,
      const bool               concentration_as_void = false,
      const bool               is_accumulative       = false)
      : InitialValues<dim>(interface_width,
                           interface_direction,
                           op_components_offset,
                           is_accumulative)
      , concentration_as_void(concentration_as_void)
    {}

  protected:
    double
    do_value(const Point<dim> &p, const unsigned int component) const final
    {
      double ret_val = 0;

      // Concentration of the CH equation
      if (component == (this->op_components_offset - 2))
        {
          std::vector<double> all_op_values(this->n_order_parameters());

          for (unsigned int op = 0; op < this->n_order_parameters(); ++op)
            all_op_values[op] = op_value(p, op);

          if (this->is_accumulative)
            {
              ret_val = std::accumulate(all_op_values.begin(),
                                        all_op_values.end(),
                                        0,
                                        [](auto a, auto b) {
                                          return std::move(a) + b;
                                        });
              ret_val = std::min(1.0, ret_val);
            }
          else
            {
              ret_val =
                *std::max_element(all_op_values.begin(), all_op_values.end());
            }

          if (concentration_as_void)
            ret_val = 1.0 - ret_val;
        }
      // Chemical potential of the CH equation
      else if (component == (this->op_components_offset - 1))
        {
          ret_val = 0;
        }
      // Order parameters of the AC equation
      else
        {
          const unsigned int order_parameter =
            component - this->op_components_offset;

          ret_val = op_value(p, order_parameter);
        }

      return ret_val;
    }

    virtual double
    op_value(const Point<dim> &p, const unsigned int order_parameter) const = 0;

  private:
    // Treat concentration term as a void param
    const bool concentration_as_void;
  };
} // namespace Sintering
