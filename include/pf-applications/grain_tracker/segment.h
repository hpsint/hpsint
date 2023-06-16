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

#include <deal.II/base/point.h>

namespace GrainTracker
{
  using namespace dealii;

  /* Segment is a part of a grain created from the previously detected cloud.
   * Segments are represented as circles with a given center, radius, measure
   * and order parameter maximum value.
   */
  template <int dim>
  class Segment
  {
  public:
    Segment(const Point<dim> &center_in,
            const double      radius_in,
            const double      measure_in,
            const double      max_value_in = 0.0)
      : center(center_in)
      , radius(radius_in)
      , measure(measure_in)
      , max_value(max_value_in)
    {}

    const Point<dim> &
    get_center() const
    {
      return center;
    }

    double
    get_radius() const
    {
      return radius;
    }

    double
    get_measure() const
    {
      return measure;
    }


    double
    get_max_value() const
    {
      return max_value;
    }

    double
    distance(const Segment<dim> &other) const
    {
      const double distance_centers = get_center().distance(other.get_center());
      const double sum_radii        = get_radius() + other.get_radius();

      const double current_distance = distance_centers - sum_radii;

      return current_distance;
    }

  protected:
    Point<dim> center;

    double radius;
    double measure;
    double max_value;
  };
} // namespace GrainTracker