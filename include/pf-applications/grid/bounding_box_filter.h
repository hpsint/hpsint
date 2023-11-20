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

#include <deal.II/base/bounding_box.h>

namespace dealii
{
  template <int dim, typename Number = double>
  class BoundingBoxFilter
  {
  public:
    enum class Position
    {
      Outside  = -1,
      Boundary = 0,
      Inside   = 1
    };

    struct Plane
    {
      Point<dim, Number> origin;
      Point<dim, Number> normal;
    };

    BoundingBoxFilter(const BoundingBox<dim, Number> &box)
      : bounding_box(box)
    {
      const auto &boundary_points = bounding_box.get_boundary_points();

      for (unsigned int d = 0; d < dim; ++d)
        {
          planes[d].origin    = boundary_points.first;
          planes[d].normal[d] = -1;

          planes[dim + d].origin    = boundary_points.second;
          planes[dim + d].normal[d] = -1;
        }
    }

    const std::array<Plane, 2 * dim> &
    get_planes() const
    {
      return planes;
    }

    Position
    position(const Point<dim> &p) const
    {
      Position point_location = Position::Outside;

      if (bounding_box.point_inside(p))
        {
          point_location = Position::Inside;

          for (unsigned int d = 0; d < dim; ++d)
            {
              if (std::abs(bounding_box.lower_bound(d) - p[d]) <
                    std::numeric_limits<Number>::epsilon() ||
                  std::abs(bounding_box.upper_bound(d) - p[d]) <
                    std::numeric_limits<Number>::epsilon())
                {
                  point_location = Position::Boundary;
                  break;
                }
            }
        }

      return point_location;
    }

    bool
    point_outside(const Point<dim> &p) const
    {
      return position(p) == Position::Outside;
    }

    bool
    point_inside(const Point<dim> &p) const
    {
      return position(p) == Position::Inside;
    }

    bool
    point_boundary(const Point<dim> &p) const
    {
      return position(p) == Position::Boundary;
    }

    bool
    point_inside_or_boundary(const Point<dim> &p) const
    {
      return position(p) != Position::Outside;
    }

    bool
    point_outside_or_boundary(const Point<dim> &p) const
    {
      return position(p) != Position::Inside;
    }

  private:
    const BoundingBox<dim, Number> bounding_box;
    std::array<Plane, 2 * dim>     planes;
  };
} // namespace dealii