#pragma once

#include "segment.h"

namespace GrainTracker
{
  /* This class represents a physical grain as a part of the domain. A single
   * grain normally consists of a single segment unless periodic boundary
   * conditions are imposed over the domain: in this case one may have multiple
   * segments. The grain id is unique but multiple grains can be assigned to the
   * same order parameter, this is the whole idea of the grain tracker feature.
   *
   * To be compatible with remapping strategy, a grain has to store:
   *
   * - grain_id to identify the grain between timesteps;
   * - order_parameter_id to know which order parameter the grain belongs to;
   * - old_order_parameter_id to know which was the previous order parameter
   * in case it has been changed.
   */
  template <int dim>
  class Grain
  {
  public:
    Grain(const unsigned int grain_id, const unsigned int order_parameter_id)
      : grain_id(grain_id)
      , order_parameter_id(order_parameter_id)
      , old_order_parameter_id(order_parameter_id)
    {}

    Grain(const unsigned int grain_id,
          const unsigned int order_parameter_id,
          const unsigned int old_order_parameter_id)
      : grain_id(grain_id)
      , order_parameter_id(order_parameter_id)
      , old_order_parameter_id(old_order_parameter_id)
    {}

    Grain()
      : grain_id(numbers::invalid_unsigned_int)
      , order_parameter_id(numbers::invalid_unsigned_int)
      , old_order_parameter_id(numbers::invalid_unsigned_int)
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
              double current_distance = this_segment.distance(other_segment);
              min_distance = std::min(current_distance, min_distance);
            }
        }

      return min_distance;
    }

    /* Check if two grains match */
    bool
    is_equal(const Grain<dim> &other) const
    {
      if (n_segments() != other.n_segments())
        {
          return false;
        }

      if (get_order_parameter_id() != other.get_order_parameter_id())
        {
          return false;
        }

      double tol_distance = 1e-10;

      for (const auto &this_segment : segments)
        {
          double min_distance = std::numeric_limits<double>::max();

          for (const auto &other_segment : other.get_segments())
            {
              double centers_distance =
                this_segment.get_center().distance(other_segment.get_center());
              min_distance = std::min(centers_distance, min_distance);
            }

          if (min_distance > tol_distance)
            {
              return false;
            }
        }

      return true;
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

    /* Set grain id. */
    void
    set_grain_id(unsigned int id)
    {
      grain_id = id;
    }

    /* Get current order parameter id. */
    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    /* Set current order parameter id.
     *
     * If this method happens to be called and the new order parameter is
     * different from the old one, that means that at the later remapping stage
     * we need to move the nodal dofs values related to the current grain from
     * the old order parameter to the new one.
     */
    void
    set_order_parameter_id(const unsigned int new_order_parameter_id)
    {
      order_parameter_id = new_order_parameter_id;
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

      max_radius = std::max(max_radius, segment.get_radius());
    }

    /* Add a grain's neighbor. Neighbors are grains having the same order
     * parameter id.
     */
    void
    add_neighbor(const Grain *neighbor)
    {
      AssertThrow(this != neighbor,
                  dealii::ExcMessage(
                    "Grain can not be added as a neighbot to itself"));
      AssertThrow(
        order_parameter_id == neighbor->get_order_parameter_id() ||
          old_order_parameter_id == neighbor->get_old_order_parameter_id(),
        dealii::ExcMessage(
          "Neighbors should have the same order parameter (current or old)."));

      neighbors.insert(neighbors.end(), neighbor);
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

    /* Get transfer buffer. A zone around the grain which will be moved to
     * another order parameter if remapping is invoked for this grain.
     */
    double
    transfer_buffer() const
    {
      return std::max(0.0, distance_to_nearest_neighbor() / 2.0);
    }

    /* Get number of segments contributing to the grain */
    unsigned int
    n_segments() const
    {
      return segments.size();
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &grain_id;
      ar &order_parameter_id;
      ar &old_order_parameter_id;
      ar &segments;
      ar &max_radius;

      // Neighbors are not serialized and transferred
    }

  private:
    unsigned int grain_id;

    unsigned int order_parameter_id;

    unsigned int old_order_parameter_id;

    std::vector<Segment<dim>> segments;

    std::set<const Grain *> neighbors;

    double max_radius{0.0};
  };
} // namespace GrainTracker
