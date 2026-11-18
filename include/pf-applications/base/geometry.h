// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2026 by the hpsint authors
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

#include <cmath>

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

namespace hpsint
{
  using namespace dealii;

  /* Translation of the point b that brings it to the periodic image closest to
   * the point a. The periods tensor holds the domain extent along the periodic
   * directions and zero along the others.
   */
  template <int dim>
  Tensor<1, dim>
  minimum_image_shift(const Point<dim>     &a,
                      const Point<dim>     &b,
                      const Tensor<1, dim> &periods)
  {
    Tensor<1, dim> shift;

    for (unsigned int d = 0; d < dim; ++d)
      if (periods[d] > 0.)
        shift[d] = -periods[d] * std::round((b[d] - a[d]) / periods[d]);

    return shift;
  }
} // namespace hpsint
