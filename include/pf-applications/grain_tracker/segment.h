#pragma once

#include <deal.II/base/point.h>

#include "cloud.h"

namespace GrainTracker
{
  template <int dim>
  class Segment
  {
  public:
    Segment(const Cloud<dim> &cloud)
    {
      double grain_volume = 0.0;
      for (const auto &cell : cloud.get_cells())
        {
          center += cell.barycenter() * cell.measure();
          grain_volume += cell.measure();
        }

      center /= grain_volume;

      // Calculate the radius as the largest distance from the centroid to the
      // center of the most distant cell
      radius = 0.0;
      for (const auto &cell : cloud.get_cells())
        {
          double dist =
            center.distance(cell.barycenter()) + cell.diameter() / 2.;
          radius = std::max(radius, dist);
        }
    }

    dealii::Point<dim>
    get_center() const
    {
      return center;
    }

    double
    get_radius() const
    {
      return radius;
    }

  protected:
    dealii::Point<dim> center;

    double radius;
  };
} // namespace GrainTracker