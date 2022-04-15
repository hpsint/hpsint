#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

#include "cell.h"

namespace GrainTracker
{
  using namespace dealii;
  
  /* Cloud is nothing by a group of cells that represent a part of a grain.
   * It works as a candidate for future grain segments. When cloud is being
   * constructed, it is associated with a certain order parameter provided via
   * its constructor.
   */
  template <int dim>
  class Cloud
  {
  public:
    Cloud()
      : order_parameter_id(dealii::numbers::invalid_unsigned_int)
    {}

    Cloud(const unsigned int order_parameter_id)
      : order_parameter_id(order_parameter_id)
    {}

    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    template <typename T>
    void
    add_cell(const T &cell_accessor)
    {
      cells.emplace_back(cell_accessor);
    }

    template <typename T>
    void
    add_edge_cell(const T &cell_accessor)
    {
      edge_cells.emplace_back(cell_accessor);
    }

    template <typename T>
    void
    add_periodic_primary_cell(const T &cell_accessor)
    {
      periodic_primary_cells.emplace_back(cell_accessor);
    }

    template <typename T>
    void
    add_periodic_secondary_cell(const T &cell_accessor)
    {
      periodic_secondary_cells.emplace_back(cell_accessor);
    }

    bool
    has_periodic_boundary() const
    {
      AssertThrow((!periodic_primary_cells.empty() &&
                   !periodic_secondary_cells.empty()) ||
                    (periodic_primary_cells.empty() &&
                     periodic_secondary_cells.empty()),
                  ExcMessage("Periodic boundary information is inconsistent"));
                  
      return !periodic_primary_cells.empty() &&
             !periodic_secondary_cells.empty();
    }

    const std::vector<Cell<dim>> &
    get_cells() const
    {
      return cells;
    }

    const std::vector<Cell<dim>> &
    get_edge_cells() const
    {
      return edge_cells;
    }

    const std::vector<Cell<dim>> &
    get_periodic_primary_cells() const
    {
      return periodic_primary_cells;
    }

    const std::vector<Cell<dim>> &
    get_periodic_secondary_cells() const
    {
      return periodic_secondary_cells;
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &cells;
      ar &edge_cells;
      ar &order_parameter_id;
    }

    void
    stitch(const Cloud &cloud)
    {
      // Append inner cells
      for (const auto &cell : cloud.get_cells())
        {
          add_cell(cell);
        }

      // Append edge cells
      for (const auto &cell : cloud.get_edge_cells())
        {
          add_edge_cell(cell);
        }
    }

    bool
    is_stitchable_with(const Cloud &cloud) const
    {
      return find_overlap(get_cells(), cloud.get_cells());
      //return find_overlap(get_edge_cells(), cloud.get_edge_cells());
    }

    bool
    is_periodic_with(const Cloud &cloud) const
    {
      return find_overlap(get_periodic_primary_cells(),
                          cloud.get_periodic_secondary_cells());
    }

  private:
    bool
    find_overlap(const std::vector<Cell<dim>> &first,
                 const std::vector<Cell<dim>> &second) const
    {
      bool overlap_found = false;

      for (const auto &cell_first : first)
        {
          for (const auto &cell_second : second)
            {
              if (cell_first.distance(cell_second) < 0.0)
                {
                  overlap_found = true;
                  break;
                }
            }

          if (overlap_found)
            {
              break;
            }
        }

      return overlap_found;
    }

    unsigned int order_parameter_id;

    std::vector<Cell<dim>> cells;
    std::vector<Cell<dim>> edge_cells;
    std::vector<Cell<dim>> periodic_primary_cells;
    std::vector<Cell<dim>> periodic_secondary_cells;
  };
} // namespace GrainTracker