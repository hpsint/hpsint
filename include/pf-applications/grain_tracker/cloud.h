#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

namespace GrainTracker
{
  template <int dim>
  class Cloud
  {
  public:
    Cloud(unsigned int oid)
      : order_parameter_id(oid)
    {}

    unsigned int
    get_order_parameter_id() const
    {
      return order_parameter_id;
    }

    void
    add_cell(const dealii::CellAccessor<dim> &cell)
    {
      cells.push_back(cell);
    }

    const std::vector<dealii::CellAccessor<dim>> &
    get_cells() const
    {
      return cells;
    }

  private:
    unsigned int order_parameter_id;

    std::vector<dealii::CellAccessor<dim>> cells;
  };
} // namespace GrainTracker