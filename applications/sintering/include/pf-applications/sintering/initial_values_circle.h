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

#include "initial_values_array.h"

namespace Sintering
{
  template <int dim>
  class InitialValuesCircle : public InitialValuesArray<dim>
  {
  public:
    InitialValuesCircle(const double       r0,
                        const double       interface_width,
                        const unsigned int n_grains,
                        const bool         minimize_order_parameters,
                        const bool         is_accumulative)
      : InitialValuesArray<dim>(r0, interface_width, is_accumulative)
    {
      const double alfa = 2 * M_PI / n_grains;

      const double h = (n_grains > 1) ? r0 / std::sin(alfa / 2.) : 0.;

      for (unsigned int ip = 0; ip < n_grains; ip++)
        {
          std::array<double, dim> scoords{{h, ip * alfa}};
          if (dim == 3)
            scoords[2] = M_PI / 2.;
          this->centers.push_back(
            dealii::GeometricUtilities::Coordinates::from_spherical<dim>(
              scoords));
        }

      if (minimize_order_parameters)
        {
          if (n_grains == 1)
            {
              this->order_parameter_to_grains[0] = {0};
            }
          else
            {
              this->order_parameter_to_grains[0];
              this->order_parameter_to_grains[1];

              for (unsigned int ip = 0; ip < n_grains; ip++)
                {
                  const unsigned int current_order_parameter = ip % 2;
                  this->order_parameter_to_grains.at(current_order_parameter)
                    .push_back(ip);
                }

              /* If the number of particles is odd, then the order parameter of
               * the last grain has to be changed to 2
               */
              if (n_grains % 2)
                {
                  const unsigned int last_grain =
                    this->order_parameter_to_grains.at(0).back();
                  this->order_parameter_to_grains.at(0).pop_back();

                  this->order_parameter_to_grains[2] = {last_grain};
                }
            }
        }
      else
        {
          for (unsigned int ip = 0; ip < n_grains; ip++)
            {
              this->order_parameter_to_grains[ip] = {ip};
            }
        }
    }
  };
} // namespace Sintering