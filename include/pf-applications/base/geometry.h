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
    BoundingBox<dim> box(center);
    box.extend(radius);

    return box;
  }
} // namespace dealii