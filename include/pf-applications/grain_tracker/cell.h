// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

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
      for (const unsigned int v : cell_accessor.vertex_indices())
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