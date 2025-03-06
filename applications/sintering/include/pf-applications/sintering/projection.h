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
      MappingQ<dim>      mapping{1};

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

      const auto &fe = background_dof_handler.get_fe();

      AssertThrow(
        fe.reference_cell().is_hyper_cube() && fe.tensor_degree() == 1,
        ExcMessage(
          "This algorithm is implemented for linear quad elements only"));

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

                        // Transfer graint tracker data to the projected grid
                        if (n_coarsening_steps == 0)
                          cell_data.material_id =
                            cell->global_active_cell_index();

                        ++vertex_counter;
                        ++vertex_numerator;
                      }
                  }

                Assert(vertex_counter == cell_data.vertices.size(),
                       ExcMessage(
                         "Number of vertices " +
                         std::to_string(vertex_counter) +
                         " added to the global set is not equal to the "
                         "number of vertices " +
                         std::to_string(cell_data.vertices.size()) +
                         " in the CellData object"));

                cells.push_back(cell_data);
              }
          }

      if (vertices.size() > 0)
        {
          projection->tria.create_triangulation(vertices, cells, subcelldata);
          projection->dof_handler.distribute_dofs(projection->fe_dg);
        }

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();

      return projection;
    }

    /* Output projection of a grain to a text file. Only spherical grain
     * representation is supported here, since we need to search for an
     * intersection between a plane and a 3-dimenshional shape, which is
     * currently implemented for spheres only.  */
    template <int dim, typename VectorType, typename Number>
    void
    output_grain_contours_projected(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      const std::string                             filename,
      const unsigned int                            n_op,
      const GrainTracker::Tracker<dim + 1, Number> &grain_tracker,
      const Point<dim + 1> &                        plane_origin,
      const Point<dim + 1> &                        plane_normal,
      const MPI_Comm &                              comm = MPI_COMM_WORLD,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      static_assert(dim == 2);

      const auto &grains = grain_tracker.get_grains();

      const auto bb = GridTools::compute_bounding_box(
        background_dof_handler.get_triangulation());

      std::vector<Number> parameters;

      // Get grain properties from the grain tracker
      unsigned int g_counter = 0;
      std::map<std::pair<unsigned int, unsigned int>, unsigned int>
                                                         grain_id_to_index;
      std::vector<std::pair<unsigned int, unsigned int>> grains_data;
      for (const auto &[g, grain] : grains)
        for (unsigned int segment_id = 0;
             segment_id < grain.get_segments().size();
             ++segment_id)
          {
            const auto &segment = grain.get_segments()[segment_id];

            const auto dist =
              (segment.get_center() - plane_origin) * plane_normal;
            const auto circle_center =
              segment.get_center() + dist * plane_normal;

            std::copy(circle_center.begin_raw(),
                      circle_center.end_raw(),
                      std::back_inserter(parameters));

            const auto circle_radius =
              (dist <= segment.get_radius()) ?
                std::sqrt(std::pow(segment.get_radius(), 2) -
                          std::pow(dist, 2)) :
                0;

            parameters.push_back(circle_radius);

            grain_id_to_index.emplace(std::make_pair(g, segment_id), g_counter);
            ++g_counter;

            grains_data.emplace_back(grain.get_grain_id(),
                                     grain.get_order_parameter_id());
          }

      std::vector<std::vector<Point<dim>>> points_local(g_counter);

      if (background_dof_handler.n_dofs() > 0)
        {
          const GridTools::MarchingCubeAlgorithm<dim,
                                                 typename VectorType::BlockType>
            mc(mapping,
               background_dof_handler.get_fe(),
               n_subdivisions,
               tolerance);

          for (unsigned int b = 0; b < n_op; ++b)
            {
              for (const auto &cell :
                   background_dof_handler.active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    auto particle_id =
                      grain_tracker.get_particle_index(b, cell->material_id());

                    if (particle_id == numbers::invalid_unsigned_int)
                      continue;

                    const auto grain_and_segment_ids =
                      grain_tracker.get_grain_and_segment(b, particle_id);

                    if (grain_and_segment_ids.first ==
                        numbers::invalid_unsigned_int)
                      continue;

                    mc.process_cell(cell,
                                    vector.block(b + 2),
                                    iso_level,
                                    points_local[grain_id_to_index.at(
                                      grain_and_segment_ids)]);
                  }
            }
        }

      internal::write_grain_contours_tex(g_counter,
                                         n_op,
                                         grains_data,
                                         bb,
                                         parameters,
                                         points_local,
                                         filename,
                                         comm);
    }

    template <int dim, typename VectorType>
    void
    output_grain_contours_projected_vtu(
      const Mapping<dim> &        mapping,
      const DoFHandler<dim> &     background_dof_handler,
      const VectorType &          vector,
      const double                iso_level,
      const std::string           filename,
      const unsigned int          n_op,
      const GrainTracker::Mapper &grain_mapper,
      const MPI_Comm &            comm           = MPI_COMM_WORLD,
      const unsigned int          n_subdivisions = 1,
      const double                tolerance      = 1e-10)
    {
      static_assert(dim == 2);

      std::function<unsigned int(const CellAccessor<dim, dim> &)>
        cell_data_extractor =
          [](const CellAccessor<dim, dim> &cell) { return cell.material_id(); };

      internal::build_write_grain_contours_vtu(mapping,
                                               background_dof_handler,
                                               vector,
                                               iso_level,
                                               filename,
                                               n_op,
                                               cell_data_extractor,
                                               grain_mapper,
                                               comm,
                                               n_subdivisions,
                                               tolerance);
    }
  } // namespace Postprocessors
} // namespace Sintering