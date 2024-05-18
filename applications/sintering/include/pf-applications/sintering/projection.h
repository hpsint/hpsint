// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi_large_count.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/base/data.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_sintering_data.h>
#include <pf-applications/sintering/postprocessors.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>
#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/grid_tools.h>

namespace Sintering
{
  namespace Postprocessors
  {
    template <int dim, typename Number>
    struct StateData
    {
      using BlockVector = std::vector<Vector<Number>>;

      Triangulation<dim> tria;
      DoFHandler<dim>    dof_handler;
      BlockVector        solution;
      FE_DGQ<dim>        fe_dg{1};

      StateData()
        : dof_handler(tria)
      {}

      StateData(unsigned int n)
        : dof_handler(tria)
        , solution(n)
      {}

      StateData(const StateData<dim, Number> &state) = delete;
    };

    template <typename VectorType, int dim = 3>
    std::unique_ptr<StateData<dim - 1, typename VectorType::value_type>>
    build_projection(const DoFHandler<dim> &background_dof_handler,
                     const VectorType &     vector,
                     const unsigned int     direction                = 2,
                     const double           location                 = 0,
                     const unsigned int     n_coarsening_steps       = 0,
                     const typename VectorType::value_type tolerance = 1e-10)
    {
      static_assert(dim == 3);

      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

      // Coarsen background mesh 1 or 2 times to reduce memory consumption

      auto vector_to_be_used                 = &vector;
      auto background_dof_handler_to_be_used = &background_dof_handler;

      parallel::distributed::Triangulation<dim> tria_copy(
        background_dof_handler.get_communicator());
      DoFHandler<dim> dof_handler_copy;
      VectorType      solution_dealii;

      if (n_coarsening_steps != 0)
        {
          coarsen_triangulation(tria_copy,
                                background_dof_handler,
                                dof_handler_copy,
                                vector,
                                solution_dealii,
                                n_coarsening_steps);

          vector_to_be_used                 = &solution_dealii;
          background_dof_handler_to_be_used = &dof_handler_copy;
        }

      std::vector<Point<dim - 1>>    vertices;
      std::vector<CellData<dim - 1>> cells;
      SubCellData                    subcelldata;

      // Data to define cross-section
      Point<dim> origin;
      origin[direction] = location;

      Point<dim> normal;
      normal[direction] = 1;

      std::vector<unsigned int> projector;
      for (unsigned int d = 0; d < dim; ++d)
        if (d != direction)
          projector.push_back(d);

      auto projection =
        std::make_unique<StateData<dim - 1, typename VectorType::value_type>>(
          vector_to_be_used->n_blocks());

      for (const auto &cell :
           background_dof_handler_to_be_used->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto ref_point       = cell->center();
            ref_point[direction] = location;

            if (cell->bounding_box().point_inside(ref_point))
              {
                CellData<dim - 1> cell_data;
                unsigned int      vertex_counter = 0;
                unsigned int vertex_numerator = projection->solution[0].size();

                for (unsigned int b = 0; b < vector_to_be_used->n_blocks(); ++b)
                  {
                    projection->solution[b].grow_or_shrink(
                      projection->solution[b].size() +
                      cell_data.vertices.size());
                  }

                // Iterate over each line of the cell
                for (unsigned int il = 0; il < cell->n_lines(); il++)
                  {
                    // Check if there are intersections with box planes
                    const auto [has_itersection, fac, p] =
                      intersect_line_plane(cell->line(il)->vertex(0),
                                           cell->line(il)->vertex(1),
                                           origin,
                                           normal);

                    if (has_itersection && std::abs(fac) < 1. + tolerance)
                      {
                        cell_data.vertices[vertex_counter] = vertex_numerator;

                        Point<dim - 1> p_proj;
                        for (unsigned int j = 0; j < projector.size(); ++j)
                          {
                            p_proj[j] = p[projector[j]];
                          }
                        vertices.push_back(p_proj);

                        // Now interpolate values
                        // DOFs correspnding to the vertices
                        const auto index0 =
                          cell->line(il)->vertex_dof_index(0, 0);
                        const auto index1 =
                          cell->line(il)->vertex_dof_index(1, 0);

                        for (unsigned int b = 0;
                             b < vector_to_be_used->n_blocks();
                             ++b)
                          {
                            // The field values associated with those DOFs
                            const auto val0 = vector.block(b)[index0];
                            const auto val1 = vector.block(b)[index1];

                            const auto val_proj = val0 + fac * (val0 - val1);

                            projection->solution[b][vertex_numerator] =
                              val_proj;
                          }

                        ++vertex_counter;
                        ++vertex_numerator;
                      }
                  }

                cells.push_back(cell_data);
              }
          }

      if (vertices.size() > 0)
        projection->tria.create_triangulation(vertices, cells, subcelldata);
      else
        GridGenerator::hyper_cube(projection->tria, -1e-6, 1e-6);

      projection->dof_handler.distribute_dofs(projection->fe_dg);

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();

      return projection;
    }
  } // namespace Postprocessors
} // namespace Sintering