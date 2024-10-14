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

#include <deal.II/base/bounding_box.h>

#include <pf-applications/sintering/csv_reader.h>
#include <pf-applications/sintering/initial_values.h>

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
  class InitialValuesMicrostructure : public InitialValues<2>
  {
    struct MicroGrain
    {
      std::vector<Point<2>> vertices;
      unsigned int          color;
      BoundingBox<2>        box;
      double                interface_width;

      // This function is base on this algo:
      // https://wrfranklin.org/Research/Short_Notes/pnpoly.html
      bool
      point_inside(const dealii::Point<2> &p) const
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

      // This function computes value for a point that is located at the
      // polygon diffuse interface, the algo is based on this discussion:
      // https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
      double
      point_value(const dealii::Point<2> &p) const
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

        if (dist_min < interface_width)
          return -1. / interface_width * dist_min + 1.;
        else
          return 0;
      }
    };

  public:
    InitialValuesMicrostructure(std::istream &stream,
                                const double  interface_width = 0.)
      : interface_width(interface_width)
    {
      unsigned int row_counter = 0;

      for (internal::CSVIterator loop(stream); loop != internal::CSVIterator();
           ++loop, ++row_counter)
        {
          if (row_counter == 0)
            {
              const auto n_grains = std::stoi(std::string((*loop)[0]));
              grains.resize(n_grains);
            }
          else if (row_counter == 1)
            {
              AssertDimension(grains.size(), loop->size());

              for (unsigned int i = 0; i < loop->size(); ++i)
                {
                  grains[i].color = std::stoi(std::string((*loop)[i]));
                  order_parameter_to_grains[grains[i].color].push_back(i);
                }
            }
          else
            {
              auto &grain = grains[row_counter - 2];

              grain.interface_width = interface_width;

              for (unsigned int i = 0; i < loop->size(); i += 2)
                {
                  grain.vertices.emplace_back(
                    std::stod(std::string((*loop)[i])),
                    std::stod(std::string((*loop)[i + 1])));
                }

              grain.vertices.push_back(grain.vertices[0]);

              grain.box = BoundingBox<2>(grain.vertices);
              grain.box.extend(interface_width);
            }
        }
    }

    double
    do_value(const dealii::Point<2> &p,
             const unsigned int      component) const final
    {
      double p_val = 0.;

      if (component > 1)
        {
          const unsigned int order_parameter = component - 2;

          std::vector<unsigned int> candidates;

          for (const auto gid : order_parameter_to_grains.at(order_parameter))
            {
              const auto &grain = grains.at(gid);

              if (grain.box.point_inside(p))
                candidates.push_back(gid);
            }

          for (const auto cadidate_id : candidates)
            if (grains[cadidate_id].point_inside(p))
              {
                p_val = 1.;
                break;
              }
            else if (interface_width > 0)
              {
                p_val = std::max(p_val, grains[cadidate_id].point_value(p));
              }
            else
              {
                p_val = 0.;
                break;
              }
        }

      return p_val;
    }

    std::pair<Point<2>, Point<2>>
    get_domain_boundaries() const final
    {
      double x_min = std::numeric_limits<double>::max();
      double y_min = std::numeric_limits<double>::max();
      double x_max = -std::numeric_limits<double>::max();
      double y_max = -std::numeric_limits<double>::max();

      for (const auto &grain : grains)
        {
          const auto &bp = grain.box.get_boundary_points();

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
        r_max =
          std::max({r_max,
                    0.5 * std::sqrt(std::pow(grain.box.side_length(0), 2) +
                                    std::pow(grain.box.side_length(1), 2))});

      return r_max;
    }

    double
    get_interface_width() const final
    {
      return interface_width;
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

  private:
    // Interface thickness
    const double interface_width;

    // Grains
    std::vector<MicroGrain> grains;

    // Map order parameters to specific grains
    std::map<unsigned int, std::vector<unsigned int>> order_parameter_to_grains;
  };
} // namespace Sintering
