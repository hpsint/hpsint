#pragma once

#include <deal.II/grid/tria_accessor.h>

#include <vector>

namespace GrainTracker
{
  template <int dim>
  class Cloud
  {
  public:
    Cloud()
      : grain_id(0)
    {}

    Cloud(unsigned int gid)
      : grain_id(gid)
    {}

    void
    set_grain_id(unsigned int gid)
    {
      grain_id = gid;
    }

    unsigned int
    get_grain_id() const
    {
      return grain_id;
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
    unsigned int grain_id;

    std::vector<dealii::CellAccessor<dim>> cells;
  };
} // namespace GrainTracker