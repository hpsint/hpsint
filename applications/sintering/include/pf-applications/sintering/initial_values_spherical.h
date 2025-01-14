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

#include <pf-applications/sintering/initial_values_ch_ac.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim>
  class InitialValuesSpherical : public InitialValuesCHAC<dim>
  {
  public:
    InitialValuesSpherical(
      const double             interface_width,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset  = 2,
      const bool               concentration_as_void = false,
      const bool               is_accumulative       = false)
      : InitialValuesCHAC<dim>(interface_width,
                               interface_direction,
                               op_components_offset,
                               concentration_as_void,
                               is_accumulative)
      , interface_offset(interface_direction == InterfaceDirection::inside ?
                           (-interface_width / 2.) :
                           (interface_direction == InterfaceDirection::outside ?
                              interface_width / 2. :
                              0.))
    {}

  protected:
    double
    is_in_sphere(const Point<dim> &point,
                 const Point<dim> &center,
                 double            rc) const
    {
      double c = 0;

      double rm  = rc - interface_offset;
      double rad = center.distance(point);

      if (rad <= rm - this->interface_width / 2.0)
        {
          c = 1;
        }
      else if (rad < rm + this->interface_width / 2.0)
        {
          double outvalue = 0.;
          double invalue  = 1.;
          double int_pos =
            (rad - rm + this->interface_width / 2.0) / this->interface_width;

          c = outvalue + (invalue - outvalue) *
                           (1.0 + std::cos(int_pos * numbers::PI)) / 2.0;
        }

      return c;
    }

  private:
    const double interface_offset;
  };
} // namespace Sintering
