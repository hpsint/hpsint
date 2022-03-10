#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

#include "cell.h"

namespace GrainTracker
{
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

    // std::vector<dealii::CellAccessor<dim>> cells;
    std::vector<Cell<dim>> cells;
  };
} // namespace GrainTracker