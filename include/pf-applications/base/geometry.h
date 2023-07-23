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

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/point.h>

namespace dealii
{
  template <int dim>
  BoundingBox<dim>
  create_bounding_box_around_point(const Point<dim> &center,
                                   const double      radius)
  {
    Point<dim> lower_left = center;
    Point<dim> top_right  = center;
    for (unsigned int d = 0; d < dim; ++d)
      {
        lower_left[d] -= radius;
        top_right[d] += radius;
      }

    return BoundingBox<dim>({lower_left, top_right});
  }
} // namespace dealii