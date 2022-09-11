#pragma once

#include <deal.II/base/point.h>

namespace GrainTracker
{
  using namespace dealii;

  /* Segment is a part of a grain created from the previously detected cloud.
   * Segments are represented as circles with a given center and radius.
   */
  template <int dim>
  class Segment
  {
  public:
    Segment(const Point<dim> &center_in, const double radius_in)
      : center(center_in)
      , radius(radius_in)
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
  };
} // namespace GrainTracker