#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

#include "cell.h"

namespace GrainTracker
{
  /* Cloud is nothing by a group of cells that represent a part of a grain.
   * It works as a candidate for future grain segments. When cloud is being
   * constructed, it is associated with a certain order parameter provided via
   * its ctor.
   */
  template <int dim>
  class Cloud
  {
  public:
    Cloud()
      : order_parameter_id(0)
    {}

    Cloud(unsigned int oid)
      : order_parameter_id(oid)
    {}

    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    void
    add_cell(const dealii::CellAccessor<dim> &cell_accessor)
    {
      cells.emplace_back(cell_accessor);
    }

    void
    add_cell(const Cell<dim> &cell)
    {
      cells.push_back(cell);
    }

    const std::vector<Cell<dim>> &
    get_cells() const
    {
      return cells;
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &cells;
      ar &order_parameter_id;
    }

  private:
    unsigned int order_parameter_id;

    std::vector<Cell<dim>> cells;
  };
} // namespace GrainTracker