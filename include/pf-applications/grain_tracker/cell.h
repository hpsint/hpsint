#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/tria_accessor.h>

#include <functional>
#include <vector>

namespace GrainTracker
{
  template <int dim>
  class Cell
  {
  public:
    Cell()
    {}

    Cell(const dealii::CellAccessor<dim> &cell_accessor)
    {
      for (unsigned int v = 0; v < cell_accessor.n_vertices(); v++)
        {
          vertices.push_back(cell_accessor.vertex(v));
        }

      // Cache cell values
      point_center     = cell_accessor.center();
      point_barycenter = cell_accessor.barycenter();
      spatial_measure  = cell_accessor.measure();
      spatial_diameter = cell_accessor.diameter();
    }

    dealii::Point<dim>
    center() const
    {
      return point_center;
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
      ar &point_center;
      ar &point_barycenter;
      ar &spatial_measure;
      ar &spatial_diameter;
      ar &vertices;
    }

    const std::vector<dealii::Point<dim>> &
    get_vertices() const
    {
      return vertices;
    }

    double
    distance(const Cell<dim> &other) const
    {
      double distance_centers = center().distance(other.center());
      double sum_radii        = (diameter() + other.diameter()) / 2.;

      double current_distance = distance_centers - sum_radii;

      return current_distance;
    }

  private:
    dealii::Point<dim> point_center;
    dealii::Point<dim> point_barycenter;

    double spatial_measure;
    double spatial_diameter;

    std::vector<dealii::Point<dim>> vertices;
  };
} // namespace GrainTracker