#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

#include "cell.h"

namespace GrainTracker
{
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

  private:
    unsigned int order_parameter_id;

    std::vector<Cell<dim>> cells;
    std::vector<Cell<dim>> edge_cells;
  };
} // namespace GrainTracker