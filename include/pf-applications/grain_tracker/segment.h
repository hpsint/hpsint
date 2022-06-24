#pragma once

#include <deal.II/base/point.h>

#include "cloud.h"

namespace GrainTracker
{
  /* Segment is a part of a grain created from the previously detected cloud.
   * Segments are represented as circles with a given center and radius.
   */
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
          const double dist =
            center.distance(cell.barycenter()) + cell.diameter() / 2.;
          radius = std::max(radius, dist);
        }
    }

    Segment()
      : radius(0.0)
    {}

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

    double
    distance(const Segment<dim> &other) const
    {
      const double distance_centers = get_center().distance(other.get_center());
      const double sum_radii        = get_radius() + other.get_radius();

      const double current_distance = distance_centers - sum_radii;

      return current_distance;
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &center;
      ar &radius;
    }

  protected:
    dealii::Point<dim> center;

    double radius;
  };
} // namespace GrainTracker