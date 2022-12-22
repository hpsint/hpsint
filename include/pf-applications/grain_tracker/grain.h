#pragma once

#include "segment.h"

namespace GrainTracker
{
  using namespace dealii;

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
    enum Dynamics
    {
      Shrinking = -1,
      None      = 0,
      Growing   = 1
    };

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
    set_grain_id(const unsigned int new_grain_id)
    {
      grain_id = new_grain_id;
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

    /* Get number of segments */
    unsigned int
    n_segments() const
    {
      return segments.size();
    }

    /* Add a new segment to the grain. */
    void
    add_segment(const Segment<dim> &segment)
    {
      segments.push_back(segment);

      max_radius = std::max(max_radius, segment.get_radius());
    }

    void
    add_segment(const Point<dim> &center, const double radius)
    {
      segments.emplace_back(center, radius);

      max_radius = std::max(max_radius, radius);
    }

    /* Add a grain's neighbor. Neighbors are grains having the same order
     * parameter id. We do need to store the complete list of neighbors,
     * we only need to know the distance to the nearest one.
     */
    void
    add_neighbor(const Grain<dim> &neighbor)
    {
      AssertThrow(this != &neighbor,
                  ExcMessage("Grain can not be added as a neighbor to itself"));
      AssertThrow(
        order_parameter_id == neighbor.get_order_parameter_id() ||
          old_order_parameter_id == neighbor.get_old_order_parameter_id(),
        ExcMessage(
          "Neighbors should have the same order parameter (current or old)."));

      distance_to_nearest_neighbor =
        std::min(distance_to_nearest_neighbor, distance(neighbor));
    }

    /* Get transfer buffer. A zone around the grain which will be moved to
     * another order parameter if remapping is invoked for this grain.
     */
    double
    transfer_buffer() const
    {
      return std::max(0.0, distance_to_nearest_neighbor / 2.0);
    }

    /* Get grain dynamics. This property says whether the grain is growing or
     * shrinking or has just been initiated.
     */
    Dynamics
    get_dynamics() const
    {
      return dynamics;
    }

    /* Set grain dynamics. The dynamics is analyzed by the user. */
    void
    set_dynamics(const Dynamics new_dynamics)
    {
      dynamics = new_dynamics;
    }

  private:
    unsigned int grain_id;

    unsigned int order_parameter_id;

    unsigned int old_order_parameter_id;

    std::vector<Segment<dim>> segments;

    double max_radius{0.0};

    double distance_to_nearest_neighbor{std::numeric_limits<double>::max()};

    Dynamics dynamics{None};
  };
} // namespace GrainTracker
