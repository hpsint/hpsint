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

#include <deal.II/grid/tria_accessor.h>

#include <type_traits>

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

    template <typename VectorizedArrayType>
    VectorizedArrayType
    filter(const Point<dim, VectorizedArrayType> &p) const
    {
      static_assert(
        std::is_same_v<typename VectorizedArrayType::value_type, Number>,
        "Type mismatch");

      const auto zeros = VectorizedArrayType(0.0);
      const auto ones  = VectorizedArrayType(1.0);

      VectorizedArrayType filter_val = ones;

      for (unsigned int d = 0; d < dim; ++d)
        {
          filter_val = compare_and_apply_mask<SIMDComparison::greater_than>(
            p[d],
            VectorizedArrayType(bounding_box.lower_bound(d)),
            filter_val,
            zeros);

          filter_val = compare_and_apply_mask<SIMDComparison::less_than>(
            p[d],
            VectorizedArrayType(bounding_box.upper_bound(d)),
            filter_val,
            zeros);
        }

      return filter_val;
    }

    bool
    intersects(const TriaAccessor<dim, dim, dim> &cell) const
    {
      unsigned int n_inside  = 0;
      unsigned int n_outside = 0;

      for (unsigned int i = 0; i < cell.n_vertices(); ++i)
        {
          const auto &point = cell.vertex(i);
          const auto  pos   = position(point);

          if (pos == Position::Outside)
            ++n_outside;
          else if (pos == Position::Inside)
            ++n_inside;
        }

      return n_inside < cell.n_vertices() && n_outside < cell.n_vertices();
    }

    const BoundingBox<dim, Number> &
    get_bounding_box() const
    {
      return bounding_box;
    }

  private:
    const BoundingBox<dim, Number> bounding_box;
    std::array<Plane, 2 * dim>     planes;
  };

  template <int dim, typename Number>
  std::tuple<bool, Number, Point<dim, Number>>
  intersect_line_plane(const Point<dim, Number> &p0,
                       const Point<dim, Number> &p1,
                       const Point<dim, Number> &p_co,
                       const Point<dim, Number> &p_no,
                       Number                    epsilon = 1e-6)
  {
    auto u = p0 - p1;

    const auto dot = p_no * u;

    if (std::abs(dot) > epsilon)
      {
        /*
         * The factor of the point between p0 -> p1 (0 - 1)
         * if 'fac' is between (-1 ... 1) the point intersects with the
         * segment.
         */
        const auto w = p0 - p_co;

        const auto fac = -(p_no * w) / dot;
        u *= fac;

        const auto res = p0 + u;

        return std::make_tuple(true, fac, res);
      }

    // The segment is parallel to plane.
    return std::make_tuple(false, 0, Point<dim, Number>());
  }
} // namespace dealii