#pragma once

#include "segment.h"

namespace GrainTracker
{
  /* This class represents a physical grain as a part of the domain. A single
   * grain normally consists of a single segment unless periodic boundary
   * conditions are imposed over the domain: in this case one may have multiple
   * segments. The grain id is unique but multiple grains can be assigned to the
   * same order parameter, this is the whole idea of the grain tracker feature.
   */
  template <int dim>
  class Grain
  {
  public:
    Grain(unsigned int gid, unsigned int oid = 0, unsigned int ooid = 0)
      : grain_id{gid}
      , order_parameter_id{oid}
      , old_order_parameter_id{ooid}
    {}

    /* This function computes the minimum distance between the segments of the
     * two grains.
     */
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

    /* Center of the grain. It is later used to identify the same grain within
     * two subsequent timesteps. If the grain consists of multiple segments,
     * then this function returns a geometric center of all grains which still
     * can be used to identify a family of segments at different timesteps.
     */
    dealii::Point<dim>
    get_center() const
    {
      return center;
    }

    /* Radius of the largest segment of the grain. Mainly used as a reference
     * value to determine the buffer zone for the grain reassignment.
     */
    double
    get_max_radius() const
    {
      return max_radius;
    }

    /* Get grain id. */
    unsigned int
    get_grain_id() const
    {
      return grain_id;
    }

    /* Get current order parameter id. */
    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    /* Set current order parameter id. */
    void
    set_order_parameter_id(unsigned int oid)
    {
      order_parameter_id = oid;
    }

    /* Get previous order parameter id. */
    unsigned int
    get_old_order_parameter_id() const
    {
      return old_order_parameter_id;
    }

    /* Get segments of the grain. */
    const std::vector<Segment<dim>> &
    get_segments() const
    {
      return segments;
    }

    /* Add a new segment to the grain. */
    void
    add_segment(const Segment<dim> &segment)
    {
      segments.push_back(segment);
      center_raw += segment.get_center();
      center = center_raw;
      center /= segments.size();

      max_radius = std::max(max_radius, segment.get_radius());
    }

    /* Add a grain's neighbor. Neighbors are grains having the same order
     * parameter id.
     */
    void
    add_neighbor(const Grain *neighbor)
    {
      if (this != neighbor &&
          order_parameter_id == neighbor->get_order_parameter_id())
        {
          neighbors.insert(neighbors.end(), neighbor);
        }
    }

    /* Get distance to the nearest neighbor. This is used to check whether any
     * of the two grains assigned to the same order parameter are getting too
     * close. If so, then for either of the two grains the order parameter has
     * to be changed.
     */
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

  private:
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