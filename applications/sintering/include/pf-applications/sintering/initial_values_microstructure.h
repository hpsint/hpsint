// ---------------------------------------------------------------------
//
// Copyright (C) 2024 - 2025 by the hpsint authors
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

#include <pf-applications/sintering/csv_reader.h>
#include <pf-applications/sintering/initial_values_ch_ac.h>

namespace Sintering
{
  using namespace dealii;

  /* This class represents custom microstructure read from a file. Works only
   * for the 2D case. The file has the following format:
   *
   * N                   - number of grains
   * op_0 op_1 ... op_N  - order parameters indices for each grain
   * x_0_0 y_0_0 x_0_1 y_0_1 ... x_0_M0 y_0_M0 - coordinates of the contour
   * M0 points of grain 0
   * ....
   * x_N_0 y_N_0 x_N_1 y_N_1 ... x_N_MN y_N_MN - coordinates of the contour
   * MN points of grain N
   *
   * The points of each contour have to be written in the clockwise or
   * counterclockwise order, the first point is not not repeated at the end.
   */
  class InitialValuesMicrostructure : public InitialValuesCHAC<2>
  {
  protected:
    class MicroSegment
    {
    public:
      template <typename Iterator>
      MicroSegment(Iterator           begin,
                   Iterator           end,
                   const double       interface_width,
                   InterfaceDirection interface_direction)
      {
        std::copy(begin, end, std::back_inserter(vertices));

        vertices.push_back(vertices[0]);
        box = BoundingBox<2>(vertices);

        if (interface_direction == InterfaceDirection::outside)
          box.extend(interface_width);
        else if (interface_direction == InterfaceDirection::middle)
          box.extend(interface_width / 2.);
      }

      // This function is base on this algo:
      // https://wrfranklin.org/Research/Short_Notes/pnpoly.html
      bool
      point_inside(const Point<2> &p) const
      {
        bool is_inside = false;

        for (unsigned int i = 0, j = vertices.size() - 1; i < vertices.size();
             j = i++)
          {
            if (((vertices[i][1] > p[1]) != (vertices[j][1] > p[1])) &&
                (p[0] < (vertices[j][0] - vertices[i][0]) *
                            (p[1] - vertices[i][1]) /
                            (vertices[j][1] - vertices[i][1]) +
                          vertices[i][0]))
              is_inside = !is_inside;
          }

        return is_inside;
      }

      bool
      point_inside_box(const Point<2> &p) const
      {
        return box.point_inside(p);
      }

      // This function computes the distance from the segment to a point,
      // the algo is based on this discussion:
      // https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
      double
      distance(const Point<2> &p) const
      {
        double dist_min = std::numeric_limits<double>::max();

        for (unsigned int i = 0; i < vertices.size() - 1; ++i)
          {
            const auto &p1 = vertices[i];
            const auto &p2 = vertices[i + 1];

            const Point<2> p1p2(p2 - p1);
            const Point<2> pp1(p - p1);

            auto r = p1p2 * pp1;

            r /= p1.distance_square(p2);

            double dist;
            if (r < 0)
              dist = p.distance(p1);
            else if (r > 1)
              dist = p2.distance(p);
            else
              dist = std::sqrt(p.distance_square(p1) -
                               r * r * p1.distance_square(p2));

            dist_min = std::min(dist, dist_min);
          }

        return dist_min;
      }

      const BoundingBox<2> &
      bounding_box() const
      {
        return box;
      }

    private:
      std::vector<Point<2>> vertices;
      BoundingBox<2>        box;
    };

    class MicroGrain
    {
    public:
      MicroGrain(
        const unsigned int color = std::numeric_limits<unsigned int>::max())
        : color(color)
      {}

      template <typename S>
      void
      add_segment(S &&segment)
      {
        segments.emplace_back(std::forward<S>(segment));
      }

      bool
      point_inside(const Point<2> &p) const
      {
        return std::find_if(segments.cbegin(),
                            segments.cend(),
                            [&p](const auto &segment) {
                              return segment.point_inside_box(p) &&
                                     segment.point_inside(p);
                            }) != segments.cend();
      }

      bool
      point_inside_box(const Point<2> &p) const
      {
        return std::find_if(segments.cbegin(),
                            segments.cend(),
                            [&p](const auto &segment) {
                              return segment.point_inside_box(p);
                            }) != segments.cend();
      }

      double
      distance_to_nearest_edge(const Point<2> &p) const
      {
        std::vector<double> distances(segments.size());

        std::transform(segments.begin(),
                       segments.end(),
                       distances.begin(),
                       [&p](const auto &segment) {
                         return segment.distance(p);
                       });

        const auto dist_min =
          *std::min_element(distances.begin(), distances.end());

        return dist_min;
      }

      BoundingBox<2>
      bounding_box() const
      {
        auto it = segments.cbegin();

        BoundingBox<2> bb(it->bounding_box());
        ++it;

        std::for_each(it, segments.cend(), [&bb](auto &segment) {
          bb.merge_with(segment.bounding_box());
        });

        return bb;
      }

      double
      max_diameter() const
      {
        double dia_max = -std::numeric_limits<double>::max();

        std::for_each(
          segments.cbegin(), segments.cend(), [&dia_max](auto &segment) {
            dia_max = std::max(
              {dia_max,
               std::sqrt(std::pow(segment.bounding_box().side_length(0), 2) +
                         std::pow(segment.bounding_box().side_length(1), 2))});
          });

        return dia_max;
      }

      void
      set_color(unsigned int new_color)
      {
        color = new_color;
      }

      unsigned int
      get_color() const
      {
        return color;
      }

    private:
      unsigned int              color;
      std::vector<MicroSegment> segments;
    };

    InitialValuesMicrostructure(
      const double             interface_width     = 0.,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset  = 2,
      const bool               concentration_as_void = false,
      const bool               is_accumulative       = false)
      : InitialValuesCHAC<2>(interface_width,
                             interface_direction,
                             op_components_offset,
                             concentration_as_void,
                             is_accumulative)
      , ref_h(interface_direction == InterfaceDirection::middle ?
                interface_width / 2. :
                interface_width)
      , ref_k(interface_direction == InterfaceDirection::middle ? 0.5 : 1.0)
      , ref_b(
          interface_direction == InterfaceDirection::middle ?
            0.5 :
            (interface_direction == InterfaceDirection::outside ? 1.0 : 0.0))
    {}

  protected:
    double
    op_value(const Point<2> &p, const unsigned int order_parameter) const final
    {
      double p_val = 0.;

      std::vector<unsigned int> candidates;

      for (const auto gid : order_parameter_to_grains.at(order_parameter))
        {
          const auto &grain = grains.at(gid);

          if (grain.point_inside_box(p))
            candidates.push_back(gid);
        }

      for (const auto cadidate_id : candidates)
        if (grains[cadidate_id].point_inside(p))
          {
            // If we are inside, then for sure no need to check other
            // candidates and we can set up p_val directly here and break
            if (interface_direction == InterfaceDirection::outside ||
                interface_width == 0)
              {
                p_val = 1.;
              }
            else
              {
                const auto dist =
                  grains[cadidate_id].distance_to_nearest_edge(p);

                p_val = (dist < ref_h) ? (ref_b + ref_k * dist / ref_h) : 1.;
              }
            break;
          }
        else if (interface_width > 0 &&
                 interface_direction != InterfaceDirection::inside)
          {
            const auto dist = grains[cadidate_id].distance_to_nearest_edge(p);

            const auto c_val =
              (dist < ref_h) ? (ref_b - ref_k * dist / ref_h) : 0.;

            p_val = std::max(p_val, c_val);
          }

      return p_val;
    }

  public:
    std::pair<Point<2>, Point<2>>
    get_domain_boundaries() const final
    {
      double x_min = std::numeric_limits<double>::max();
      double y_min = std::numeric_limits<double>::max();
      double x_max = -std::numeric_limits<double>::max();
      double y_max = -std::numeric_limits<double>::max();

      for (const auto &grain : grains)
        {
          const auto bp = grain.bounding_box().get_boundary_points();

          x_min = std::min(x_min, bp.first[0]);
          y_min = std::min(y_min, bp.first[1]);
          x_max = std::max(x_max, bp.second[0]);
          y_max = std::max(y_max, bp.second[1]);
        }

      return std::make_pair(Point<2>(x_min, y_min), Point<2>(x_max, y_max));
    }

    double
    get_r_max() const final
    {
      double r_max = -std::numeric_limits<double>::max();

      for (const auto &grain : grains)
        r_max = std::max({r_max, 0.5 * grain.max_diameter()});

      return r_max;
    }

    unsigned int
    n_order_parameters() const final
    {
      return order_parameter_to_grains.size();
    }

    unsigned int
    n_particles() const final
    {
      return grains.size();
    }

  protected:
    // Reference values for the linear approximation
    const double ref_h;
    const double ref_k;
    const double ref_b;

    // Grains
    std::vector<MicroGrain> grains;

    // Map order parameters to specific grains
    std::map<unsigned int, std::vector<unsigned int>> order_parameter_to_grains;
  };
} // namespace Sintering
