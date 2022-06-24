#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/tria_accessor.h>

#include <functional>
#include <vector>

namespace GrainTracker
{
  /* The very first implementation used dealii::CellAccessor instead of this
   * class. But then it was observed that dealii::CellAccessor can not be
   * serialized and thus transferred via MPI calls. So this simplified object
   * mimics some basic functionality of dealii::CellAccessor used in the grain
   * tracker implementation. Only selected functions are needed for querying the
   * geometry related data: cell size, center and distance to other cells.
   */
  template <int dim>
  class Cell
  {
  public:
    Cell() = default;

    Cell(const dealii::CellAccessor<dim> &cell_accessor)
    {
      // Cache cell values
      point_barycenter = cell_accessor.barycenter();
      spatial_measure  = cell_accessor.measure();
      spatial_diameter = cell_accessor.diameter();
    }

    dealii::Point<dim>
    barycenter() const
    {
      return point_barycenter;
    }

    double
    measure() const
    {
      return spatial_measure;
    }

    double
    diameter() const
    {
      return spatial_diameter;
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &point_barycenter;
      ar &spatial_measure;
      ar &spatial_diameter;
    }

    double
    distance(const Cell<dim> &other) const
    {
      double distance_centers = barycenter().distance(other.barycenter());
      double sum_radii        = (diameter() + other.diameter()) / 2.;

      double current_distance = distance_centers - sum_radii;

      return current_distance;
    }

  private:
    dealii::Point<dim> point_barycenter;

    double spatial_measure;
    double spatial_diameter;
  };
} // namespace GrainTracker