#pragma once

#include "segment.h"

namespace GrainTracker
{
  template <int dim>
  class Grain
  {
  public:
    Grain(unsigned int gid, unsigned int oid = 0, unsigned int ooid = 0)
      : grain_id{gid}
      , order_parameter_id{oid}
      , old_order_parameter_id{ooid}
    {}

    double
    distance(const Grain<dim> &other) const
    {
      double min_distance = std::numeric_limits<double>::max();
      for (const auto &this_segment : segments)
        {
          for (const auto &other_segment : other.get_segments())
            {
              double distance_centers =
                this_segment.get_center().distance(other_segment.get_center());
              double sum_radii =
                this_segment.get_radius() + other_segment.get_radius();
              double current_distance = distance_centers - sum_radii;
              min_distance = std::min(current_distance, min_distance);
            }
        }

      return min_distance;
    }

    dealii::Point<dim>
    get_center() const
    {
      return center;
    }

    double
    get_max_radius() const
    {
      return max_radius;
    }

    unsigned int
    get_grain_id() const
    {
      return grain_id;
    }

    void
    set_grain_id(unsigned int gid)
    {
      grain_id = gid;
    }

    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    void
    set_order_parameter_id(unsigned int oid)
    {
      order_parameter_id = oid;
    }

    unsigned int
    get_old_order_parameter_id() const
    {
      return old_order_parameter_id;
    }

    const std::vector<Segment<dim>> &
    get_segments() const
    {
      return segments;
    }

    void
    add_segment(const Segment<dim> &segment)
    {
      segments.push_back(segment);
      center_raw += segment.get_center();
      center = center_raw;
      center /= segments.size();

      max_radius = std::max(max_radius, segment.get_radius());
    }

    void
    add_neighbor(const Grain *neighbor)
    {
      if (this != neighbor &&
          order_parameter_id == neighbor->get_order_parameter_id())
        {
          neighbors.insert(neighbors.end(), neighbor);
        }
    }

    double
    distance_to_nearest_neighbor() const
    {
      double dist = std::numeric_limits<double>::max();

      for (const auto nb : neighbors)
        {
          dist = std::min(dist, distance(*nb));
        }

      return dist;
    }

  protected:
    dealii::Point<dim> center;

    dealii::Point<dim> center_raw;

    unsigned int grain_id;

    unsigned int order_parameter_id;

    unsigned int old_order_parameter_id;

    std::vector<Segment<dim>> segments;

    std::set<const Grain *> neighbors;

    double max_radius{0.0};
  };
} // namespace GrainTracker