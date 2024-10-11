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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi_large_count.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/base/data.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_sintering_coupled_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>
#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/bounding_box_filter.h>
#include <pf-applications/grid/grid_tools.h>

namespace dealii
{
  template <int dim, int spacedim>
  class MyDataOut : public DataOut<dim, spacedim>
  {
  public:
    void
    write_vtu_in_parallel(
      const std::string &         filename,
      const MPI_Comm &            comm,
      const DataOutBase::VtkFlags vtk_flags = DataOutBase::VtkFlags()) const
    {
      const unsigned int myrank = Utilities::MPI::this_mpi_process(comm);

      std::ofstream ss_out(filename);

      if (myrank == 0) // header
        {
          std::stringstream ss;
          DataOutBase::write_vtu_header(ss, vtk_flags);
          ss_out << ss.rdbuf();
        }

      if (true) // main
        {
          const auto &                  patches      = this->get_patches();
          const types::global_dof_index my_n_patches = patches.size();
          const types::global_dof_index global_n_patches =
            Utilities::MPI::sum(my_n_patches, comm);

          std::stringstream ss;
          if (my_n_patches > 0 || (global_n_patches == 0 && myrank == 0))
            DataOutBase::write_vtu_main(patches,
                                        this->get_dataset_names(),
                                        this->get_nonscalar_data_ranges(),
                                        vtk_flags,
                                        ss);

          const auto temp = Utilities::MPI::gather(comm, ss.str(), 0);

          if (myrank == 0)
            for (const auto &i : temp)
              ss_out << i;
        }

      if (myrank == 0) // footer
        {
          std::stringstream ss;
          DataOutBase::write_vtu_footer(ss);
          ss_out << ss.rdbuf();
        }
    }
  };
} // namespace dealii

namespace Sintering
{
  namespace Postprocessors
  {
    template <int dim, int spacedim>
    using SurfaceDataOut =
#ifdef DISABLE_MPI_IO_SURFACE_OUTPUT
      MyDataOut<dim, spacedim>;
#else
      DataOut<dim, spacedim>;
#endif

    namespace internal
    {
      template <int dim, typename VectorType>
      bool
      build_grain_boundaries_mesh(
        Triangulation<dim - 1, dim> &                 tria,
        const Mapping<dim> &                          mapping,
        const DoFHandler<dim> &                       background_dof_handler,
        const VectorType &                            vector,
        const double                                  iso_level,
        const unsigned int                            n_grains,
        const double                                  gb_lim             = 0.14,
        const unsigned int                            n_coarsening_steps = 0,
        std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
        const unsigned int                            n_subdivisions = 1,
        const double                                  tolerance      = 1e-10)
      {
        using Number = typename VectorType::value_type;

        const bool has_ghost_elements = vector.has_ghost_elements();

        if (has_ghost_elements == false)
          vector.update_ghost_values();

        // step 0) coarsen background mesh 1 or 2 times to reduce memory
        // consumption
        auto vector_to_be_used                 = &vector;
        auto background_dof_handler_to_be_used = &background_dof_handler;

        parallel::distributed::Triangulation<dim> tria_copy(
          background_dof_handler.get_communicator());
        DoFHandler<dim> dof_handler_copy;
        VectorType      vector_coarsened;

        if (n_coarsening_steps != 0)
          {
            coarsen_triangulation(tria_copy,
                                  background_dof_handler,
                                  dof_handler_copy,
                                  vector,
                                  vector_coarsened,
                                  n_coarsening_steps);

            vector_to_be_used                 = &vector_coarsened;
            background_dof_handler_to_be_used = &dof_handler_copy;
          }

        if (box_filter)
          {
            // Copy vector if not done before
            if (n_coarsening_steps == 0)
              {
                vector_coarsened = vector;
                vector_coarsened.update_ghost_values();
                vector_to_be_used = &vector_coarsened;
              }

            auto only_order_params =
              vector_coarsened.create_view(2, 2 + n_grains);

            filter_mesh_withing_bounding_box(*background_dof_handler_to_be_used,
                                             *only_order_params,
                                             iso_level,
                                             box_filter);
          }

        const auto &fe = background_dof_handler_to_be_used->get_fe();
        FEPointEvaluation<1, dim> fe_evaluation(mapping, fe, update_values);

        // step 1) create surface mesh
        std::vector<Point<dim>>        vertices;
        std::vector<CellData<dim - 1>> cells;
        SubCellData                    subcelldata;

        const GridTools::MarchingCubeAlgorithm<dim,
                                               typename VectorType::BlockType>
          mc(mapping,
             background_dof_handler_to_be_used->get_fe(),
             n_subdivisions,
             tolerance);

        Vector<Number> values_i(
          background_dof_handler_to_be_used->get_fe().n_dofs_per_cell());
        Vector<Number> values_j(
          background_dof_handler_to_be_used->get_fe().n_dofs_per_cell());
        Vector<Number> gb(
          background_dof_handler_to_be_used->get_fe().n_dofs_per_cell());

        for (const auto &cell :
             background_dof_handler_to_be_used->active_cell_iterators())
          if (cell->is_locally_owned())
            {
              for (unsigned int i = 0; i < n_grains; ++i)
                {
                  cell->get_dof_values(vector_to_be_used->block(2 + i),
                                       values_i);

                  bool i_upper = false;
                  bool i_lower = false;
                  for (const auto i_val : values_i)
                    {
                      if (i_val > iso_level)
                        i_upper = true;
                      if (i_val < iso_level)
                        i_lower = true;
                    }

                  gb = 0;
                  if (i_upper && i_lower)
                    {
                      bool has_others = false;

                      for (unsigned int j = 0; j < n_grains; ++j)
                        {
                          if (i == j)
                            continue;

                          cell->get_dof_values(vector_to_be_used->block(2 + j),
                                               values_j);

                          gb += values_j;

                          bool j_upper = false;
                          bool j_lower = false;
                          for (const auto j_val : values_j)
                            {
                              if (j_val > iso_level)
                                j_upper = true;
                              if (j_val < iso_level)
                                j_lower = true;
                            }

                          if (j_upper && j_lower)
                            has_others = true;
                        }

                      gb.scale(values_i);

                      const bool has_strong_gb =
                        std::any_of(gb.begin(),
                                    gb.end(),
                                    [gb_lim](const auto &val) {
                                      return val > gb_lim;
                                    });

                      const bool is_gb_candidate =
                        (has_strong_gb || has_others);

                      if (is_gb_candidate)
                        {
                          std::vector<CellData<dim - 1>> local_cells;

                          mc.process_cell(cell,
                                          vector_to_be_used->block(2 + i),
                                          iso_level,
                                          vertices,
                                          local_cells);

                          std::vector<Point<dim>> real_centroids;
                          for (const auto &new_cell : local_cells)
                            {
                              Point<dim> centoroid;
                              for (const auto &vertex_id : new_cell.vertices)
                                centoroid += vertices[vertex_id];
                              centoroid /= new_cell.vertices.size();

                              real_centroids.push_back(centoroid);
                            }
                          std::vector<Point<dim>> unit_centroids(
                            real_centroids.size());

                          mapping.transform_points_real_to_unit_cell(
                            cell, real_centroids, unit_centroids);

                          fe_evaluation.reinit(cell, unit_centroids);

                          ArrayView<Number> gb_via_array_view(&gb[0],
                                                              gb.size());
                          fe_evaluation.evaluate(gb_via_array_view,
                                                 EvaluationFlags::values);

                          for (unsigned int i = 0; i < local_cells.size(); ++i)
                            if (fe_evaluation.get_value(i) > gb_lim)
                              cells.push_back(local_cells[i]);
                        }
                    }
                }
            }

        if (vertices.size() > 0 && cells.size() > 0)
          tria.create_triangulation(vertices, cells, subcelldata);

        if (has_ghost_elements == false)
          vector.zero_out_ghost_values();

        return vertices.size() > 0 && cells.size() > 0;
      }

      template <int dim, typename Number>
      void
      write_grain_contours_tex(
        const unsigned int                                        n_grains,
        const unsigned int                                        n_op,
        const std::vector<std::pair<unsigned int, unsigned int>> &grains_data,
        const BoundingBox<dim> &                                  bb,
        const std::vector<Number> &                               parameters,
        const std::vector<std::vector<Point<dim>>> &              points_local,
        const std::string                                         filename,
        MPI_Comm comm = MPI_COMM_WORLD)
      {
        /* File format (data and length):
        problem dimensionality         - dim
        number of grains               - N
        number of order_parameters     - M
        grain indices                  - array[N]
        grain order parameters         - array[N]
        BB bottom left point           - array[dim]
        BB top right point             - array[dim]
        properties (center and radius) - array[(dim+1)*N]
        particle_0                     - 1
        points_0                       - array[...]
        ...
        particle_N                     - 1
        points_N                       - array[...]
        */

        MPI_Info info;
        int      ierr = MPI_Info_create(&info);

        AssertThrowMPI(ierr);

        MPI_File fh;
        ierr = MPI_File_open(
          comm, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &fh);
        AssertThrow(ierr == MPI_SUCCESS, ExcFileNotOpen(filename));

        ierr = MPI_File_set_size(fh, 0); // delete the file contents
        AssertThrowMPI(ierr);
        ierr = MPI_Barrier(comm);
        AssertThrowMPI(ierr);
        ierr = MPI_Info_free(&info);
        AssertThrowMPI(ierr);

        // Define header size so we can broadcast later
        unsigned int header_size = 0;

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            std::stringstream ss;

            ss << dim << std::endl;
            ss << n_grains << std::endl;
            ss << n_op << std::endl;

            for (const auto &entry : grains_data)
              ss << entry.first << " ";
            ss << std::endl;

            for (const auto &entry : grains_data)
              ss << entry.second << " ";
            ss << std::endl;

            for (unsigned int d = 0; d < dim; ++d)
              ss << bb.get_boundary_points().first[d] << " ";
            ss << std::endl;

            for (unsigned int d = 0; d < dim; ++d)
              ss << bb.get_boundary_points().second[d] << " ";
            ss << std::endl;

            for (const auto &i : parameters)
              ss << i << " ";
            ss << std::endl;

            header_size = ss.str().size();

            ierr =
              Utilities::MPI::LargeCount::File_write_at_c(fh,
                                                          0,
                                                          ss.str().c_str(),
                                                          header_size,
                                                          MPI_CHAR,
                                                          MPI_STATUS_IGNORE);
            AssertThrowMPI(ierr);
          }

        ierr = MPI_Bcast(&header_size, 1, MPI_UNSIGNED, 0, comm);
        AssertThrowMPI(ierr);

        // Write points per rank to independent streams
        std::stringstream ss;
        for (unsigned int i = 0; i < n_grains; ++i)
          if (!points_local[i].empty())
            {
              ss << i << std::endl;
              for (const auto &point : points_local[i])
                ss << point << " ";
              ss << std::endl;
            }

        // Use prefix sum to find specific offset to write at
        const std::uint64_t size_on_proc = ss.str().size();
        std::uint64_t       prefix_sum   = 0;
        ierr                             = MPI_Exscan(
          &size_on_proc, &prefix_sum, 1, MPI_UINT64_T, MPI_SUM, comm);
        AssertThrowMPI(ierr);

        // Locate specific offset for each processor
        const MPI_Offset offset =
          static_cast<MPI_Offset>(header_size) + prefix_sum;

        ierr =
          Utilities::MPI::LargeCount::File_write_at_all_c(fh,
                                                          offset,
                                                          ss.str().c_str(),
                                                          ss.str().size(),
                                                          MPI_CHAR,
                                                          MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);
      }

      template <int dim, typename VectorType>
      void
      build_write_grain_contours_vtu(
        const Mapping<dim> &   mapping,
        const DoFHandler<dim> &background_dof_handler,
        const VectorType &     vector,
        const double           iso_level,
        const std::string      filename,
        const unsigned int     n_op,
        std::function<unsigned int(const CellAccessor<dim, dim> &)>
          cell_data_extractor,
        std::optional<std::reference_wrapper<const GrainTracker::Mapper>>
                           grain_mapper   = {},
        const unsigned int n_subdivisions = 1,
        const double       tolerance      = 1e-10)
      {
        std::vector<Point<dim>>        vertices;
        std::vector<CellData<dim - 1>> cells;
        SubCellData                    subcelldata;

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
                  const unsigned int old_size = cells.size();

                  mc.process_cell(
                    cell, vector.block(b + 2), iso_level, vertices, cells);

                  for (unsigned int i = old_size; i < cells.size(); ++i)
                    {
                      if (grain_mapper && !grain_mapper->get().empty())
                        {
                          const auto particle_id_for_op =
                            grain_mapper->get().get_particle_index(
                              b, cell_data_extractor(*cell));

                          if (particle_id_for_op !=
                              numbers::invalid_unsigned_int)
                            cells[i].material_id =
                              grain_mapper->get()
                                .get_grain_and_segment(b, particle_id_for_op)
                                .first;
                        }

                      cells[i].manifold_id = b;
                    }
                }
          }

        Triangulation<dim - 1, dim> tria;

        if (vertices.size() > 0)
          tria.create_triangulation(vertices, cells, subcelldata);
        else
          GridGenerator::hyper_cube(tria, -1e-6, 1e-6);

        Vector<float> vector_grain_id(tria.n_active_cells());
        Vector<float> vector_order_parameter_id(tria.n_active_cells());

        if (vertices.size() > 0)
          {
            for (const auto &cell : tria.active_cell_iterators())
              {
                vector_grain_id[cell->active_cell_index()] =
                  cell->material_id();
                vector_order_parameter_id[cell->active_cell_index()] =
                  cell->manifold_id();
              }
            tria.reset_all_manifolds();
          }
        else
          {
            vector_grain_id           = -1.0; // initialized with dummy value
            vector_order_parameter_id = -1.0;
          }

        Vector<float> vector_rank(tria.n_active_cells());
        vector_rank = Utilities::MPI::this_mpi_process(
          background_dof_handler.get_communicator());

        SurfaceDataOut<dim - 1, dim> data_out;
        data_out.attach_triangulation(tria);
        data_out.add_data_vector(vector_grain_id, "grain_id");
        data_out.add_data_vector(vector_order_parameter_id,
                                 "order_parameter_id");
        data_out.add_data_vector(vector_rank, "subdomain");

        data_out.build_patches();
        data_out.write_vtu_in_parallel(
          filename, background_dof_handler.get_communicator());
      }
    } // namespace internal

    template <int dim, typename VectorType, typename Number>
    void
    output_grain_contours(
      const Mapping<dim> &                      mapping,
      const DoFHandler<dim> &                   background_dof_handler,
      const VectorType &                        vector,
      const double                              iso_level,
      const std::string                         filename,
      const unsigned int                        n_op,
      const GrainTracker::Tracker<dim, Number> &grain_tracker,
      const unsigned int                        n_subdivisions = 1,
      const double                              tolerance      = 1e-10)
    {
      (void)mapping;


      const auto comm = background_dof_handler.get_communicator();

      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

      const auto &grains = grain_tracker.get_grains();

      const auto n_grains = grains.size();

      const auto bb = GridTools::compute_bounding_box(
        background_dof_handler.get_triangulation());

      std::vector<Number> parameters((dim + 1) * n_grains, 0);

      // Get grain properties from the grain tracker, assume 1 segment per grain
      unsigned int                         g_counter = 0;
      std::map<unsigned int, unsigned int> grain_id_to_index;
      for (const auto &[g, grain] : grains)
        {
          Assert(grain.get_segments().size() == 1, ExcNotImplemented());

          const auto &segment = grain.get_segments()[0];

          for (unsigned int d = 0; d < dim; ++d)
            parameters[g_counter * (dim + 1) + d] = segment.get_center()[d];

          parameters[g_counter * (dim + 1) + dim] = segment.get_radius();

          grain_id_to_index.emplace(g, g_counter);
          ++g_counter;
        }

      std::vector<std::vector<Point<dim>>> points_local(n_grains);

      const GridTools::MarchingCubeAlgorithm<dim,
                                             typename VectorType::BlockType>
        mc(mapping, background_dof_handler.get_fe(), n_subdivisions, tolerance);

      for (unsigned int b = 0; b < n_op; ++b)
        {
          for (const auto &cell :
               background_dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                auto particle_id = grain_tracker.get_particle_index(
                  b, cell->global_active_cell_index());

                if (particle_id == numbers::invalid_unsigned_int)
                  continue;

                const auto grain_id =
                  grain_tracker.get_grain_and_segment(b, particle_id).first;

                if (grain_id == numbers::invalid_unsigned_int)
                  continue;

                mc.process_cell(cell,
                                vector.block(b + 2),
                                iso_level,
                                points_local[grain_id_to_index.at(grain_id)]);
              }
        }

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();

      std::vector<std::pair<unsigned int, unsigned int>> grains_data;

      for (const auto &entry : grains)
        grains_data.emplace_back(entry.second.get_grain_id(),
                                 entry.second.get_order_parameter_id());

      internal::write_grain_contours_tex(n_grains,
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
    output_grain_contours_vtu(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      const std::string                             filename,
      const unsigned int                            n_grains,
      const GrainTracker::Mapper &                  grain_mapper,
      const unsigned int                            n_coarsening_steps = 0,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

      // step 0) coarsen background mesh 1 or 2 times to reduce memory
      // consumption

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

      if (box_filter)
        {
          // Copy vector if not done before
          if (n_coarsening_steps == 0)
            {
              solution_dealii = vector;
              solution_dealii.update_ghost_values();
              vector_to_be_used = &solution_dealii;
            }

          auto only_order_params = solution_dealii.create_view(2, 2 + n_grains);

          filter_mesh_withing_bounding_box(*background_dof_handler_to_be_used,
                                           *only_order_params,
                                           iso_level,
                                           box_filter);
        }

      std::function<unsigned int(const CellAccessor<dim, dim> &)>
        cell_data_extractor = [](const CellAccessor<dim, dim> &cell) {
          return cell.global_active_cell_index();
        };

      std::optional<std::reference_wrapper<const GrainTracker::Mapper>>
        opt_grain_mapper;
      if (n_coarsening_steps == 0)
        opt_grain_mapper = grain_mapper;

      internal::build_write_grain_contours_vtu(
        mapping,
        *background_dof_handler_to_be_used,
        *vector_to_be_used,
        iso_level,
        filename,
        n_grains,
        cell_data_extractor,
        opt_grain_mapper,
        n_subdivisions,
        tolerance);

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();
    }

    template <int dim, typename VectorType>
    void
    output_grain_boundaries_vtu(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      const std::string                             filename,
      const unsigned int                            n_grains,
      const double                                  gb_lim             = 0.14,
      const unsigned int                            n_coarsening_steps = 0,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      Triangulation<dim - 1, dim> tria;

      const bool tria_not_empty =
        internal::build_grain_boundaries_mesh(tria,
                                              mapping,
                                              background_dof_handler,
                                              vector,
                                              iso_level,
                                              n_grains,
                                              gb_lim,
                                              n_coarsening_steps,
                                              box_filter,
                                              n_subdivisions,
                                              tolerance);

      if (!tria_not_empty)
        GridGenerator::hyper_cube(tria, -1e-6, 1e-6);

      // step 2) output mesh
      SurfaceDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());
    }

    template <int dim, typename VectorType>
    void
    output_concentration_contour_vtu(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      const std::string                             filename,
      const unsigned int                            n_coarsening_steps = 0,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

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

      if (box_filter)
        {
          // Copy vector if not done before
          if (n_coarsening_steps == 0)
            {
              solution_dealii = vector;
              solution_dealii.update_ghost_values();
              vector_to_be_used = &solution_dealii;
            }

          filter_mesh_withing_bounding_box(*background_dof_handler_to_be_used,
                                           solution_dealii,
                                           iso_level,
                                           box_filter);
        }


      // step 1) create surface mesh
      std::vector<Point<dim>>        vertices;
      std::vector<CellData<dim - 1>> cells;
      SubCellData                    subcelldata;

      const GridTools::MarchingCubeAlgorithm<dim,
                                             typename VectorType::BlockType>
        mc(mapping,
           background_dof_handler_to_be_used->get_fe(),
           n_subdivisions,
           tolerance);

      for (const auto &cell :
           background_dof_handler_to_be_used->active_cell_iterators())
        if (cell->is_locally_owned())
          mc.process_cell(
            cell, vector_to_be_used->block(0), iso_level, vertices, cells);

      Triangulation<dim - 1, dim> tria;

      if (vertices.size() > 0)
        tria.create_triangulation(vertices, cells, subcelldata);
      else
        GridGenerator::hyper_cube(tria, -1e-6, 1e-6);

      Vector<float> vector_rank(tria.n_active_cells());
      vector_rank = Utilities::MPI::this_mpi_process(
        background_dof_handler.get_communicator());

      // step 2) output mesh
      SurfaceDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(vector_rank, "subdomain");

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();
    }

    template <int dim, typename VectorType>
    typename VectorType::value_type
    compute_surface_area(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      const auto &concentration = vector.block(0);

      const bool has_ghost_elements = concentration.has_ghost_elements();

      if (has_ghost_elements == false)
        concentration.update_ghost_values();

      std::vector<Point<dim>>        vertices;
      std::vector<CellData<dim - 1>> cells;
      SubCellData                    subcelldata;

      const GridTools::MarchingCubeAlgorithm<dim,
                                             typename VectorType::BlockType>
        mc(mapping, background_dof_handler.get_fe(), n_subdivisions, tolerance);

      for (const auto &cell : background_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          mc.process_cell(cell, concentration, iso_level, vertices, cells);

      typename VectorType::value_type surf_area = 0;
      if (vertices.size() > 0)
        {
          Triangulation<dim - 1, dim> tria;
          tria.create_triangulation(vertices, cells, subcelldata);

          for (const auto &cell : tria.active_cell_iterators())
            if (cell->is_locally_owned() &&
                (!box_filter ||
                 box_filter->point_inside_or_boundary(cell->center())))
              surf_area += cell->measure();
        }
      surf_area =
        Utilities::MPI::sum(surf_area,
                            background_dof_handler.get_communicator());

      if (has_ghost_elements == false)
        concentration.zero_out_ghost_values();

      return surf_area;
    }

    template <int dim, typename VectorType>
    typename VectorType::value_type
    compute_grain_boundaries_area(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       background_dof_handler,
      const VectorType &                            vector,
      const double                                  iso_level,
      const unsigned int                            n_grains,
      const double                                  gb_lim         = 0.14,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const double                                  tolerance      = 1e-10)
    {
      Triangulation<dim - 1, dim> tria;

      const unsigned int                            n_coarsening_steps = 0;
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter_mesh = nullptr;

      const bool tria_not_empty =
        internal::build_grain_boundaries_mesh(tria,
                                              mapping,
                                              background_dof_handler,
                                              vector,
                                              iso_level,
                                              n_grains,
                                              gb_lim,
                                              n_coarsening_steps,
                                              box_filter_mesh,
                                              n_subdivisions,
                                              tolerance);

      typename VectorType::value_type gb_area = 0;
      if (tria_not_empty)
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() &&
              (!box_filter ||
               box_filter->point_inside_or_boundary(cell->center())))
            gb_area += cell->measure();

      gb_area =
        Utilities::MPI::sum(gb_area, background_dof_handler.get_communicator());
      gb_area *= 0.5;

      return gb_area;
    }


    template <int dim, typename VectorType>
    void
    estimate_overhead(const Mapping<dim> &   mapping,
                      const DoFHandler<dim> &background_dof_handler,
                      const VectorType &     vector,
                      const bool             output_mesh = false)
    {
      using Number = typename VectorType::value_type;

      const auto comm = background_dof_handler.get_communicator();

      const std::int64_t n_active_cells_0 =
        background_dof_handler.get_triangulation().n_global_active_cells();
      std::int64_t n_active_cells_1 = 0;

      if (output_mesh)
        {
          DataOut<dim> data_out;
          data_out.attach_triangulation(
            background_dof_handler.get_triangulation());
          data_out.build_patches(mapping);
          data_out.write_vtu_in_parallel("reduced_mesh.0.vtu", comm);
        }

      {
        std::vector<unsigned int> counters(
          background_dof_handler.get_triangulation().n_active_cells(), 0);

        for (unsigned int b = 0; b < vector.n_blocks() - 2; ++b)
          {
            Vector<Number> values(
              background_dof_handler.get_fe().n_dofs_per_cell());
            for (const auto &cell :
                 background_dof_handler.active_cell_iterators())
              {
                if (cell->is_locally_owned() == false)
                  continue;

                cell->get_dof_values(vector.block(b + 2), values);

                if (values.linfty_norm() > 0.01)
                  counters[cell->active_cell_index()]++;
              }
          }

        unsigned int max_value =
          *std::max_element(counters.begin(), counters.end());
        max_value = Utilities::MPI::max(max_value, comm);

        std::vector<unsigned int> max_values(max_value, 0);

        for (const auto i : counters)
          if (i != 0)
            max_values[i - 1]++;

        Utilities::MPI::sum(max_values, comm, max_values);

        ConditionalOStream pcout(std::cout,
                                 Utilities::MPI::this_mpi_process(comm) == 0);

        pcout << "Max grains per cell: " << max_value << " (";

        pcout << (1) << ": " << max_values[0];
        for (unsigned int i = 1; i < max_values.size(); ++i)
          pcout << ", " << (i + 1) << ": " << max_values[i];

        pcout << ")" << std::endl;
      }

      for (unsigned int b = 0; b < vector.n_blocks() - 2; ++b)
        {
          parallel::distributed::Triangulation<dim> tria_copy(comm);
          DoFHandler<dim>                           dof_handler_copy;
          VectorType                                solution_dealii;

          tria_copy.copy_triangulation(
            background_dof_handler.get_triangulation());
          dof_handler_copy.reinit(tria_copy);
          dof_handler_copy.distribute_dofs(
            background_dof_handler.get_fe_collection());

          // 1) copy solution so that it has the right ghosting
          const auto partitioner =
            std::make_shared<Utilities::MPI::Partitioner>(
              dof_handler_copy.locally_owned_dofs(),
              DoFTools::extract_locally_relevant_dofs(dof_handler_copy),
              comm);

          solution_dealii.reinit(vector.n_blocks());

          for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
            {
              solution_dealii.block(b).reinit(partitioner);
              solution_dealii.block(b).copy_locally_owned_data_from(
                vector.block(b));
            }

          solution_dealii.update_ghost_values();

          unsigned int n_active_cells = tria_copy.n_global_active_cells();

          while (true)
            {
              // 2) mark cells for refinement
              Vector<Number> values(
                dof_handler_copy.get_fe().n_dofs_per_cell());
              for (const auto &cell : dof_handler_copy.active_cell_iterators())
                {
                  if (cell->is_locally_owned() == false ||
                      cell->refine_flag_set())
                    continue;

                  cell->get_dof_values(solution_dealii.block(b + 2), values);

                  if (values.linfty_norm() <= 0.05)
                    cell->set_coarsen_flag();
                }

              // 3) perform interpolation and initialize data structures
              tria_copy.prepare_coarsening_and_refinement();

              parallel::distributed::
                SolutionTransfer<dim, typename VectorType::BlockType>
                  solution_trans(dof_handler_copy);

              std::vector<const typename VectorType::BlockType *>
                solution_dealii_ptr(solution_dealii.n_blocks());
              for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
                solution_dealii_ptr[b] = &solution_dealii.block(b);

              solution_trans.prepare_for_coarsening_and_refinement(
                solution_dealii_ptr);

              tria_copy.execute_coarsening_and_refinement();

              dof_handler_copy.distribute_dofs(
                background_dof_handler.get_fe_collection());

              const auto partitioner =
                std::make_shared<Utilities::MPI::Partitioner>(
                  dof_handler_copy.locally_owned_dofs(),
                  DoFTools::extract_locally_relevant_dofs(dof_handler_copy),
                  comm);

              for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
                solution_dealii.block(b).reinit(partitioner);

              std::vector<typename VectorType::BlockType *> solution_ptr(
                solution_dealii.n_blocks());
              for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
                solution_ptr[b] = &solution_dealii.block(b);

              solution_trans.interpolate(solution_ptr);
              solution_dealii.update_ghost_values();

              if (n_active_cells == tria_copy.n_global_active_cells())
                break;

              n_active_cells = tria_copy.n_global_active_cells();
            }

          n_active_cells_1 += tria_copy.n_global_active_cells();

          if (output_mesh)
            {
              DataOut<dim> data_out;
              data_out.attach_triangulation(tria_copy);
              data_out.build_patches(mapping);
              data_out.write_vtu_in_parallel("reduced_mesh." +
                                               std::to_string(b + 1) + ".vtu",
                                             comm);
            }
        }

      ConditionalOStream pcout(std::cout,
                               Utilities::MPI::this_mpi_process(comm) == 0);

      pcout << "Estimation of mesh overhead: "
            << std::to_string((n_active_cells_0 * vector.n_blocks()) * 100 /
                                (n_active_cells_1 + 2 * n_active_cells_0) -
                              100)
            << "%" << std::endl
            << std::endl;
    }



    namespace internal
    {
      template <int dim, typename BlockVectorType, typename Number>
      unsigned int
      run_flooding(
        const typename DoFHandler<dim>::cell_iterator &cell,
        const BlockVectorType &                        solution,
        LinearAlgebra::distributed::Vector<Number> &   particle_ids,
        const unsigned int                             id,
        const double                                   threshold_upper = 0.8,
        const double invalid_particle_id                               = -1.0,
        std::shared_ptr<const BoundingBoxFilter<dim>> box_filter = nullptr)
      {
        if (cell->has_children())
          {
            unsigned int counter = 0;

            for (const auto &child : cell->child_iterators())
              counter += run_flooding<dim>(child,
                                           solution,
                                           particle_ids,
                                           id,
                                           threshold_upper,
                                           invalid_particle_id,
                                           box_filter);

            return counter;
          }

        if (cell->is_locally_owned() == false ||
            (box_filter && box_filter->point_outside(cell->barycenter())))
          return 0;

        const auto particle_id = particle_ids[cell->global_active_cell_index()];

        if (particle_id != invalid_particle_id)
          return 0; // cell has been visited

        Vector<double> values(cell->get_fe().n_dofs_per_cell());

        if (false /* TODO */)
          {
            for (unsigned int b = 2; b < solution.n_blocks(); ++b)
              {
                cell->get_dof_values(solution.block(b), values);

                if (values.linfty_norm() >= threshold_upper)
                  return 0;
              }
          }
        else
          {
            cell->get_dof_values(solution.block(0), values);

            if (values.linfty_norm() >= threshold_upper)
              return 0;
          }

        particle_ids[cell->global_active_cell_index()] = id;

        unsigned int counter = 1;

        for (const auto face : cell->face_indices())
          if (cell->at_boundary(face) == false)
            counter += run_flooding<dim>(cell->neighbor(face),
                                         solution,
                                         particle_ids,
                                         id,
                                         threshold_upper,
                                         invalid_particle_id,
                                         box_filter);

        return counter;
      }

      template <int dim, typename VectorType>
      std::tuple<LinearAlgebra::distributed::Vector<double>,
                 std::vector<unsigned int>,
                 unsigned int>
      detect_pores(
        const DoFHandler<dim> &dof_handler,
        const VectorType &     solution,
        const double           invalid_particle_id               = -1.0,
        const double           threshold_upper                   = 0.8,
        std::shared_ptr<const BoundingBoxFilter<dim>> box_filter = nullptr)
      {
        const auto comm = dof_handler.get_communicator();

        LinearAlgebra::distributed::Vector<double> particle_ids(
          dof_handler.get_triangulation()
            .global_active_cell_index_partitioner()
            .lock());

        // step 1) run flooding and determine local particles and give them
        // local ids
        particle_ids = invalid_particle_id;

        unsigned int counter = 0;
        unsigned int offset  = 0;

        const bool has_ghost_elements = solution.has_ghost_elements();

        if (has_ghost_elements == false)
          solution.update_ghost_values();

        for (const auto &cell : dof_handler.active_cell_iterators())
          if (run_flooding<dim>(cell,
                                solution,
                                particle_ids,
                                counter,
                                threshold_upper,
                                invalid_particle_id,
                                box_filter) > 0)
            counter++;

        if (has_ghost_elements == false)
          solution.zero_out_ghost_values();

        // step 2) determine the global number of locally determined particles
        // and give each one an unique id by shifting the ids
        MPI_Exscan(&counter, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

        for (auto &particle_id : particle_ids)
          if (particle_id != invalid_particle_id)
            particle_id += offset;

        // step 3) get particle ids on ghost cells and figure out if local
        // particles and ghost particles might be one particle
        particle_ids.update_ghost_values();

        auto local_connectivity = GrainTracker::build_local_connectivity(
          dof_handler, particle_ids, counter, offset, invalid_particle_id);

        // step 4) based on the local-ghost information, figure out all
        // particles on all processes that belong togher (unification ->
        // clique), give each clique an unique id, and return mapping from the
        // global non-unique ids to the global ids
        auto local_to_global_particle_ids =
          GrainTracker::perform_distributed_stitching_via_graph(
            comm, local_connectivity);

        return std::make_tuple(std::move(particle_ids),
                               std::move(local_to_global_particle_ids),
                               offset);
      }
    } // namespace internal

    template <int dim, typename VectorType>
    void
    output_porosity(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       dof_handler,
      const VectorType &                            solution,
      const std::string                             output,
      const double                                  threshold_upper = 0.8,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter      = nullptr)
    {
      const double invalid_particle_id = -1.0; // TODO

      const auto tria = dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &dof_handler.get_triangulation());

      AssertThrow(tria, ExcNotImplemented());

      // Detect pores and assign ids
      const auto [particle_ids, local_to_global_particle_ids, offset] =
        internal::detect_pores(dof_handler,
                               solution,
                               invalid_particle_id,
                               threshold_upper,
                               box_filter);

      // Output pores to VTK
      Vector<double> cell_to_id(tria->n_active_cells());

      for (const auto &cell :
           dof_handler.get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto particle_id =
              particle_ids[cell->global_active_cell_index()];

            if (particle_id == invalid_particle_id)
              cell_to_id[cell->active_cell_index()] = invalid_particle_id;
            else
              cell_to_id[cell->active_cell_index()] =
                local_to_global_particle_ids
                  [static_cast<unsigned int>(particle_id) - offset];
          }

      DataOut<dim> data_out;

      const auto get_valid_cell = [&](const auto cell_in) {
        auto cell = cell_in;

        while (cell != tria->end())
          {
            if (cell->is_active() && cell->is_locally_owned() &&
                cell_to_id[cell->active_cell_index()] != invalid_particle_id)
              break;

            ++cell;
          }

        return cell;
      };

      const auto next_cell = [&](const auto &, const auto cell_in) {
        auto cell = cell_in;
        cell++;

        return get_valid_cell(cell);
      };

      const auto first_cell = [&](const auto &tria) {
        return get_valid_cell(tria.begin());
      };

      data_out.set_cell_selection(first_cell, next_cell);

      data_out.attach_triangulation(dof_handler.get_triangulation());
      data_out.add_data_vector(cell_to_id, "ids");
      data_out.build_patches(mapping);
      data_out.write_vtu_in_parallel(output, dof_handler.get_communicator());
    }

    /* The function outputs the contours of the pores, i.e. the void regions
     * where mass concentration equals to zero (distinctly from particles/grains
     * where concentration equals to 1). Furthermore, since usually the
     * simulation domains are constructed such that there is always a void
     * region surrounding the particles assembly, this function attempts to
     * detect that region and exclude it from the output. */
    template <int dim, typename VectorType>
    void
    output_porosity_contours_vtu(
      const Mapping<dim> &                          mapping,
      const DoFHandler<dim> &                       dof_handler,
      const VectorType &                            solution,
      const double                                  iso_level,
      const std::string                             output,
      const unsigned int                            n_coarsening_steps = 0,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter     = nullptr,
      const unsigned int                            n_subdivisions = 1,
      const bool                                    smooth         = true,
      const double                                  tolerance      = 1e-10)
    {
      const auto comm = dof_handler.get_communicator();

      const double invalid_pore_id = -1.0;

      // We set up the upper bound this way to ensure that all the cells that
      // could contribute to the later construction of isocontours get captured
      // as voids.
      const double threshold_upper = std::min(1.1 * iso_level, 0.99);

      // Detect pores and assign ids
      const auto [pore_ids, local_to_global_pore_ids, offset] =
        internal::detect_pores(
          dof_handler, solution, invalid_pore_id, threshold_upper, box_filter);

      std::set<unsigned int> unique_boundary_pores_ids;

      // Eliminate pores touching the domain boundary. This improves readability
      // of the rendered picture in 3D but if there is a big pore going through
      // the microstructure, that can be observed at the beginning of the
      // sintering, then, unfortunately, it will get eliminated too. One should
      // keep this side effect in mind.
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          const auto pore_id = pore_ids[cell->global_active_cell_index()];

          if (pore_id == invalid_pore_id)
            continue;

          for (const auto face : cell->face_indices())
            if (cell->at_boundary(face))
              {
                unique_boundary_pores_ids.insert(
                  local_to_global_pore_ids[static_cast<unsigned int>(pore_id) -
                                           offset]);
                break;
              }
        }

      // We convert a set to a vector since boost can not serializes sets
      std::vector<unsigned int> global_boundary_pores_ids(
        unique_boundary_pores_ids.begin(), unique_boundary_pores_ids.end());

      const auto global_boundary_pores_temp =
        Utilities::MPI::gather(comm, global_boundary_pores_ids, 0);

      std::set<unsigned int> all_unique_boundary_pores_ids;
      for (auto &boundary_pores : global_boundary_pores_temp)
        std::copy(boundary_pores.begin(),
                  boundary_pores.end(),
                  std::inserter(all_unique_boundary_pores_ids,
                                all_unique_boundary_pores_ids.end()));

      // We convert a set to a vector since boost can not serializes sets
      std::vector<unsigned int> all_global_boundary_pores_ids(
        all_unique_boundary_pores_ids.begin(),
        all_unique_boundary_pores_ids.end());

      all_global_boundary_pores_ids =
        Utilities::MPI::broadcast(comm, all_global_boundary_pores_ids, 0);

      std::unordered_set<unsigned int> boundary_pores(
        all_global_boundary_pores_ids.begin(),
        all_global_boundary_pores_ids.end());

      // Build a vector for MCA, we need only one block. We simply set the
      // vector values to 1 if a cell belongs to a pore that does not touch the
      // domain boundary. An alternative solution would be to process quantity
      // (1-c) but that generates sometimes not desirable output when the outer
      // void has to be eliminated. So this choice generates slightly less
      // smooth but more physically representative surface contours. However,
      // the option with smoother pores is also left as available since it
      // provides better pictures if a bounding box filter was used.
      VectorType pores_data(1);
      const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(dof_handler),
        dof_handler.get_communicator());

      pores_data.block(0).reinit(partitioner);

      if (smooth)
        {
          // Use quantity (1-c)
          pores_data.block(0).copy_locally_owned_data_from(solution.block(0));
          pores_data.block(0) *= -1.0;
          for (auto &v : pores_data.block(0))
            v += 1.0;
        }
      else
        {
          // Use data from the pores info
          pores_data.block(0) = 0;
        }

      pores_data.update_ghost_values();

      Vector<typename VectorType::value_type> values(
        dof_handler.get_fe().n_dofs_per_cell());
      values = (smooth ? 0.0 : 1.0);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          const auto pore_id = pore_ids[cell->global_active_cell_index()];

          if (pore_id == invalid_pore_id)
            continue;

          const auto global_pore_id =
            local_to_global_pore_ids[static_cast<unsigned int>(pore_id) -
                                     offset];

          if ((boundary_pores.find(global_pore_id) != boundary_pores.end() &&
               smooth) ||
              (boundary_pores.find(global_pore_id) == boundary_pores.end() &&
               !smooth))
            cell->set_dof_values(values, pores_data.block(0));
        }

      // This required for the MPI case for the non-smooth version
      if (!smooth)
        {
          pores_data.compress(VectorOperation::add);

          for (auto &v : pores_data.block(0))
            v = std::min(v, 1.0);
        }

      output_concentration_contour_vtu(mapping,
                                       dof_handler,
                                       pores_data,
                                       iso_level,
                                       output,
                                       n_coarsening_steps,
                                       box_filter,
                                       n_subdivisions,
                                       tolerance);
    }

    template <int dim, typename VectorType>
    void
    output_porosity_stats(
      const DoFHandler<dim> &                       dof_handler,
      const VectorType &                            solution,
      const std::string                             output,
      const double                                  threshold_upper = 0.8,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter      = nullptr)
    {
      const double invalid_particle_id = -1.0; // TODO

      const auto comm = dof_handler.get_communicator();

      // Detect pores and assign ids
      auto [particle_ids, local_to_global_particle_ids, offset] =
        internal::detect_pores(dof_handler,
                               solution,
                               invalid_particle_id,
                               threshold_upper,
                               box_filter);

      const unsigned int n_pores =
        GrainTracker::number_of_stitched_particles(local_to_global_particle_ids,
                                                   comm);

      GrainTracker::switch_to_global_indices(particle_ids,
                                             local_to_global_particle_ids,
                                             offset,
                                             invalid_particle_id);

      const auto [pores_centers, pores_measures] =
        GrainTracker::compute_particles_info(dof_handler,
                                             particle_ids,
                                             n_pores,
                                             invalid_particle_id);

      const auto [pores_radii, pores_remotes] =
        GrainTracker::compute_particles_radii(
          dof_handler, particle_ids, pores_centers, false, invalid_particle_id);
      (void)pores_remotes;

      if (Utilities::MPI::this_mpi_process(comm) != 0)
        return;

      TableHandler table;

      std::vector<std::string> labels{"x", "y", "z"};

      for (unsigned int i = 0; i < n_pores; ++i)
        {
          table.add_value("id", i);
          table.add_value("measure", pores_measures[i]);
          table.add_value("radius", pores_radii[i]);

          for (unsigned int d = 0; d < dim; ++d)
            table.add_value(labels[d], pores_centers[i][d]);
        }

      // Output to file
      std::stringstream ss;
      table.write_text(ss);

      std::ofstream out_file(output);
      out_file << ss.rdbuf();
      out_file.close();
    }

    template <int dim, typename Number>
    void
    write_bounding_box(const BoundingBox<dim, Number> &bb,
                       const Mapping<dim> &            mapping,
                       const DoFHandler<dim> &         dof_handler,
                       const std::string               output)
    {
      Triangulation<dim> tria;
      GridGenerator::hyper_rectangle(tria,
                                     bb.get_boundary_points().first,
                                     bb.get_boundary_points().second);

      DataOut<dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.build_patches(mapping);
      data_out.write_vtu_in_parallel(output, dof_handler.get_communicator());
    }

    template <int dim, typename VectorType>
    BoundingBox<dim, typename VectorType::value_type>
    estimate_shrinkage(const Mapping<dim> &   mapping,
                       const DoFHandler<dim> &dof_handler,
                       const VectorType &     solution,
                       const unsigned int     n_intervals = 10)
    {
      const double threshold = 0.5 - 1e-2;
      const double rel_tol   = 1e-3;

      FEValues<dim> fe_values(mapping,
                              dof_handler.get_fe(),
                              dof_handler.get_fe().get_unit_support_points(),
                              update_quadrature_points);

      const auto bb_tria = dealii::GridTools::compute_bounding_box(
        dof_handler.get_triangulation());

      std::vector<typename VectorType::value_type> min_values(dim);
      std::vector<typename VectorType::value_type> max_values(dim);

      for (unsigned int d = 0; d < dim; ++d)
        {
          min_values[d] = bb_tria.get_boundary_points().second[d];
          max_values[d] = bb_tria.get_boundary_points().first[d];
        }

      using CellPtr = TriaIterator<DoFCellAccessor<dim, dim, false>>;

      std::vector<std::pair<CellPtr, double>> min_cells(
        dim, std::make_pair(CellPtr(), 0));
      std::vector<std::pair<CellPtr, double>> max_cells(
        dim, std::make_pair(CellPtr(), 0));

      Vector<typename VectorType::value_type> values;

      const bool has_ghost_elements = solution.has_ghost_elements();

      if (has_ghost_elements == false)
        solution.update_ghost_values();

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          fe_values.reinit(cell);

          values.reinit(fe_values.dofs_per_cell);

          cell->get_dof_values(solution.block(0), values);

          if (std::any_of(values.begin(),
                          values.end(),
                          [threshold](const auto &val) {
                            return val > threshold;
                          }) &&
              std::any_of(values.begin(),
                          values.end(),
                          [threshold](const auto &val) {
                            return val < threshold;
                          }))
            {
              const double c_norm  = values.linfty_norm();
              const double abs_tol = rel_tol * cell->diameter();

              for (unsigned int d = 0; d < dim; ++d)
                {
                  const auto cell_coord = cell->center()[d];

                  const auto dist_min =
                    (min_cells[d].first.state() != IteratorState::invalid) ?
                      std::abs(cell_coord - min_cells[d].first->center()[d]) :
                      0.;

                  if (min_cells[d].first.state() == IteratorState::invalid ||
                      (dist_min < abs_tol && c_norm > min_cells[d].second) ||
                      (dist_min > abs_tol &&
                       cell_coord < min_cells[d].first->center()[d]))
                    {
                      min_cells[d].first  = cell;
                      min_cells[d].second = c_norm;
                    }

                  const auto dist_max =
                    (max_cells[d].first.state() != IteratorState::invalid) ?
                      std::abs(cell_coord - max_cells[d].first->center()[d]) :
                      0.;

                  if (max_cells[d].first.state() == IteratorState::invalid ||
                      (dist_max < abs_tol && c_norm > max_cells[d].second) ||
                      (dist_max > abs_tol &&
                       cell_coord > max_cells[d].first->center()[d]))
                    {
                      max_cells[d].first  = cell;
                      max_cells[d].second = c_norm;
                    }
                }

              for (const auto q : fe_values.quadrature_point_indices())
                {
                  if (values[q] > threshold)
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        min_values[d] =
                          std::min(min_values[d],
                                   fe_values.quadrature_point(q)[d]);
                        max_values[d] =
                          std::max(max_values[d],
                                   fe_values.quadrature_point(q)[d]);
                      }
                }
            }
        }

      // Generate refined quadrature
      std::vector<dealii::Point<1>> points(n_intervals - 1);

      // The end points are dropped since the support points of the cells have
      // been already analyzed and the result is already in min/max_values
      for (unsigned int i = 0; i < n_intervals - 1; ++i)
        points[i][0] = 1. / n_intervals * (i + 1);
      Quadrature<1>   quad_1d(points);
      Quadrature<dim> quad_refined(quad_1d);

      FEValues<dim> fe_values_refined(mapping,
                                      dof_handler.get_fe(),
                                      quad_refined,
                                      update_quadrature_points | update_values);

      std::vector<typename VectorType::value_type> values_refined(
        fe_values_refined.n_quadrature_points);

      for (unsigned int d = 0; d < dim; ++d)
        {
          if (min_cells[d].first.state() == IteratorState::valid)
            {
              fe_values_refined.reinit(min_cells[d].first);
              fe_values_refined.get_function_values(solution.block(0),
                                                    values_refined);

              for (const auto q : fe_values_refined.quadrature_point_indices())
                {
                  if (values_refined[q] > threshold)
                    {
                      min_values[d] =
                        std::min(min_values[d],
                                 fe_values_refined.quadrature_point(q)[d]);
                    }
                }
            }

          if (max_cells[d].first.state() == IteratorState::valid)
            {
              fe_values_refined.reinit(max_cells[d].first);
              fe_values_refined.get_function_values(solution.block(0),
                                                    values_refined);

              for (const auto q : fe_values_refined.quadrature_point_indices())
                {
                  if (values_refined[q] > threshold)
                    {
                      max_values[d] =
                        std::max(max_values[d],
                                 fe_values_refined.quadrature_point(q)[d]);
                    }
                }
            }
        }

      if (has_ghost_elements == false)
        solution.zero_out_ghost_values();

      Utilities::MPI::min(min_values,
                          dof_handler.get_communicator(),
                          min_values);
      Utilities::MPI::max(max_values,
                          dof_handler.get_communicator(),
                          max_values);

      Point<dim> left_bb, right_bb;

      for (unsigned int d = 0; d < dim; ++d)
        {
          left_bb[d]  = min_values[d];
          right_bb[d] = max_values[d];
        }

      BoundingBox<dim, typename VectorType::value_type> bb({left_bb, right_bb});

      return bb;
    }



    template <int dim, typename VectorType>
    void
    estimate_shrinkage(const Mapping<dim> &   mapping,
                       const DoFHandler<dim> &dof_handler,
                       const VectorType &     solution,
                       const std::string      output,
                       const unsigned int     n_intervals = 10)
    {
      const auto bb =
        estimate_shrinkage(mapping, dof_handler, solution, n_intervals);

      write_bounding_box(bb, mapping, dof_handler, output);
    }

    void
    write_table(const TableHandler &table,
                const double        t,
                const MPI_Comm &    comm,
                const std::string   save_path)
    {
      if (Utilities::MPI::this_mpi_process(comm) != 0)
        return;

      const bool is_new = (t == 0);

      std::stringstream ss;
      table.write_text(ss);

      std::string line;

      std::ofstream ofs;
      ofs.open(save_path,
               is_new ? std::ofstream::out | std::ofstream::trunc :
                        std::ofstream::app);

      // Get header
      std::getline(ss, line);

      // Write header if we only start writing
      if (is_new)
        ofs << line << std::endl;

      // Take the data itself
      std::getline(ss, line);

      ofs << line << std::endl;
      ofs.close();
    }

    namespace internal
    {
      template <int dim, typename BlockVectorType>
      void
      do_estimate_mesh_quality(
        const DoFHandler<dim> &dof_handler,
        const BlockVectorType &solution,
        std::function<void(const typename BlockVectorType::value_type qval,
                           const DoFCellAccessor<dim, dim, false> &)>
          store_result)
      {
        const bool has_ghost_elements = solution.has_ghost_elements();

        if (has_ghost_elements == false)
          solution.update_ghost_values();

        Vector<typename BlockVectorType::value_type> values(
          dof_handler.get_fe().n_dofs_per_cell());

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            typename BlockVectorType::value_type delta_cell = 0;

            for (unsigned int b = 0; b < solution.n_blocks(); ++b)
              {
                cell->get_dof_values(solution.block(b), values);

                const auto order_parameter_min =
                  *std::min_element(values.begin(), values.end());
                const auto order_parameter_max =
                  *std::max_element(values.begin(), values.end());

                const auto delta = order_parameter_max - order_parameter_min;

                delta_cell = std::max(delta, delta_cell);
              }

            store_result(1. - delta_cell, *cell);
          }

        if (has_ghost_elements == false)
          solution.zero_out_ghost_values();
      }

      template <int dim, typename BlockVectorType>
      void
      do_output_mesh_quality(
        const Mapping<dim> &                          mapping,
        const DoFHandler<dim> &                       dof_handler,
        const BlockVectorType &                       solution,
        const std::string                             output,
        Vector<typename BlockVectorType::value_type> &quality)
      {
        const auto callback =
          [&quality](const typename BlockVectorType::value_type qval,
                     const DoFCellAccessor<dim, dim, false> &   cell) {
            quality[cell.active_cell_index()] = qval;
          };

        internal::do_estimate_mesh_quality<dim>(dof_handler,
                                                solution,
                                                callback);

        DataOut<dim> data_out;
        data_out.attach_triangulation(dof_handler.get_triangulation());
        data_out.add_data_vector(quality, "quality");
        data_out.build_patches(mapping);
        data_out.write_vtu_in_parallel(output, dof_handler.get_communicator());
      }
    } // namespace internal

    /* Output mesh quality: 0 - low, 1 - high */
    template <int dim, typename BlockVectorType>
    void
    output_mesh_quality(const Mapping<dim> &   mapping,
                        const DoFHandler<dim> &dof_handler,
                        const BlockVectorType &solution,
                        const std::string      output)
    {
      Vector<typename BlockVectorType::value_type> quality(
        dof_handler.get_triangulation().n_active_cells());

      internal::do_output_mesh_quality(
        mapping, dof_handler, solution, output, quality);
    }

    /* Output mesh quality and return its min: 0 - low, 1 - high */
    template <int dim, typename BlockVectorType>
    typename BlockVectorType::value_type
    output_mesh_quality_and_min(const Mapping<dim> &   mapping,
                                const DoFHandler<dim> &dof_handler,
                                const BlockVectorType &solution,
                                const std::string      output)
    {
      Vector<typename BlockVectorType::value_type> quality(
        dof_handler.get_triangulation().n_active_cells());

      internal::do_output_mesh_quality(
        mapping, dof_handler, solution, output, quality);

      const auto min_quality =
        *std::min_element(quality.begin(), quality.end());

      return Utilities::MPI::min(min_quality, dof_handler.get_communicator());
    }

    /* Estimate min mesh quality: 0 - low, 1 - high */
    template <int dim, typename BlockVectorType>
    typename BlockVectorType::value_type
    estimate_mesh_quality_min(const DoFHandler<dim> &dof_handler,
                              const BlockVectorType &solution)
    {
      typename BlockVectorType::value_type quality = 1.;

      const auto callback =
        [&quality](const typename BlockVectorType::value_type qval,
                   const DoFCellAccessor<dim, dim, false> &   cell) {
          (void)cell;
          quality = std::min(quality, qval);
        };

      internal::do_estimate_mesh_quality<dim>(dof_handler, solution, callback);

      quality = Utilities::MPI::min(quality, dof_handler.get_communicator());

      return quality;
    }

    /* Build scalar quantities to compute */
    template <int dim, typename VectorizedArrayType>
    auto
    build_domain_quantities_evaluators(
      const std::vector<std::string> &                       labels,
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data)
    {
      using QuantityCallback = std::function<
        VectorizedArrayType(const VectorizedArrayType *,
                            const Tensor<1, dim, VectorizedArrayType> *,
                            const unsigned int)>;

      std::vector<std::string>      q_labels;
      std::vector<QuantityCallback> q_evaluators;

      for (const auto &qty : labels)
        {
          if (qty == "solid_vol")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)gradient;
                (void)n_grains;

                return value[0];
              });
          else if (qty == "surf_area")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)gradient;
                (void)n_grains;

                return value[0] * (1.0 - value[0]);
              });
          else if (qty == "gb_area")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)gradient;

                VectorizedArrayType eta_ij_sum = 0.0;
                for (unsigned int i = 0; i < n_grains; ++i)
                  for (unsigned int j = i + 1; j < n_grains; ++j)
                    eta_ij_sum += value[2 + i] * value[2 + j];

                return eta_ij_sum;
              });
          else if (qty == "avg_grain_size")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)gradient;

                VectorizedArrayType eta_i2_sum = 0.0;
                for (unsigned int i = 0; i < n_grains; ++i)
                  eta_i2_sum += value[2 + i] * value[2 + i];

                return eta_i2_sum;
              });
          else if (qty == "surf_area_nrm")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)gradient;
                (void)n_grains;

                VectorizedArrayType c_int(1.0);
                c_int = compare_and_apply_mask<SIMDComparison::less_than>(
                  value[0],
                  VectorizedArrayType(0.45),
                  VectorizedArrayType(0.0),
                  c_int);
                c_int = compare_and_apply_mask<SIMDComparison::greater_than>(
                  value[0],
                  VectorizedArrayType(0.55),
                  VectorizedArrayType(0.0),
                  c_int);

                return c_int;
              });
          else if (qty == "free_energy")
            q_evaluators.emplace_back(
              [&sintering_data](
                const VectorizedArrayType *                value,
                const Tensor<1, dim, VectorizedArrayType> *gradient,
                const unsigned int                         n_grains) {
                VectorizedArrayType energy(0.0);

                std::vector<VectorizedArrayType> etas(n_grains);
                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  {
                    etas[ig] = value[2 + ig];
                    energy += gradient[2 + ig].norm_square();
                  }
                energy *= 0.5 * sintering_data.kappa_p;

                const auto &c      = value[0];
                const auto &c_grad = gradient[0];
                energy += 0.5 * sintering_data.kappa_c * c_grad.norm_square();

                energy += sintering_data.free_energy.f(c, etas);

                return energy;
              });
          else if (qty == "bulk_energy")
            q_evaluators.emplace_back(
              [&sintering_data](
                const VectorizedArrayType *                value,
                const Tensor<1, dim, VectorizedArrayType> *gradient,
                const unsigned int                         n_grains) {
                (void)gradient;

                const VectorizedArrayType &c = value[0];

                std::vector<VectorizedArrayType> etas(n_grains);
                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  etas[ig] = value[2 + ig];

                return sintering_data.free_energy.f(c, etas);
              });
          else if (qty == "interface_energy")
            q_evaluators.emplace_back(
              [&sintering_data](
                const VectorizedArrayType *                value,
                const Tensor<1, dim, VectorizedArrayType> *gradient,
                const unsigned int                         n_grains) {
                (void)value;

                VectorizedArrayType energy(0.0);

                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  energy += gradient[2 + ig].norm_square();
                energy *= 0.5 * sintering_data.kappa_p;

                const auto &c_grad = gradient[0];
                energy += 0.5 * sintering_data.kappa_c * c_grad.norm_square();

                return energy;
              });
          else if (qty == "order_params")
            for (unsigned int i = 0; i < MAX_SINTERING_GRAINS; ++i)
              {
                // The number of order parameters can vary so we will output the
                // maximum number of them. The unused order parameters will be
                // simply filled with zeros.
                q_labels.push_back("op_" + std::to_string(i));

                q_evaluators.emplace_back(
                  [i](const VectorizedArrayType *                value,
                      const Tensor<1, dim, VectorizedArrayType> *gradient,
                      const unsigned int                         n_grains) {
                    (void)gradient;

                    return i < n_grains ? value[2 + i] : 0.;
                  });
              }
          else if (qty == "control_vol")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType *                value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient,
                 const unsigned int                         n_grains) {
                (void)value;
                (void)gradient;
                (void)n_grains;

                return VectorizedArrayType(1.);
              });
          else
            AssertThrow(false,
                        ExcMessage("Invalid domain integral provided: " + qty));

          if (qty != "order_params")
            q_labels.push_back(qty);
        }

      AssertDimension(q_labels.size(), q_evaluators.size());

      return std::make_tuple(q_labels, q_evaluators);
    }

    template <int dim,
              typename NonLinearOperator,
              typename VectorType,
              typename VectorizedArrayType,
              typename Number>
    void
    output_grains_stats(
      const Mapping<dim> &                      mapping,
      const DoFHandler<dim> &                   dof_handler,
      const NonLinearOperator &                 sintering_operator,
      const GrainTracker::Tracker<dim, Number> &grain_tracker,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
        &               advection_mechanism,
      const VectorType &solution,
      const std::string output)
    {
      const auto comm = dof_handler.get_communicator();

      const bool has_ghost_elements = solution.has_ghost_elements();

      if (has_ghost_elements == false)
        solution.update_ghost_values();

      // We assume that a grain contains a single segment
      constexpr unsigned int segment_id = 0;

      Tensor<1, dim, Number> dummy_velocities;
      for (unsigned int d = 0; d < dim; ++d)
        dummy_velocities[d] = std::numeric_limits<double>::quiet_NaN();

      // Two different ways to compute velocities
      std::function<Tensor<1, dim, Number>(const unsigned int)> get_velocity;
      if constexpr (std::is_base_of_v<
                      SinteringOperatorCoupledBase<dim,
                                                   Number,
                                                   VectorizedArrayType,
                                                   NonLinearOperator>,
                      NonLinearOperator>)
        {
          std::vector<Point<dim>> evaluation_points;

          std::array<std::vector<Number>, dim> displacements;

          std::map<unsigned int, unsigned int> grain_id_to_eval_id;
          unsigned int                         counter = 0;

          for (const auto &[grain_id, grain] : grain_tracker.get_grains())
            {
              grain_id_to_eval_id[grain_id] = counter++;
              evaluation_points.push_back(
                grain.get_segments()[segment_id].get_center());
            }

          // set up cache manually
          Utilities::MPI::RemotePointEvaluation<dim, dim> rpe;
          rpe.reinit(evaluation_points,
                     dof_handler.get_triangulation(),
                     mapping);

          // use the cache
          for (unsigned int b = solution.n_blocks() - dim, d = 0;
               b < solution.n_blocks();
               ++b, ++d)
            {
              displacements[d] =
                VectorTools::point_values<1>(rpe,
                                             dof_handler,
                                             solution.block(b));
            }

          get_velocity =
            [displacements       = std::move(displacements),
             grain_id_to_eval_id = std::move(grain_id_to_eval_id),
             dummy_velocities    = std::move(dummy_velocities),
             dt = sintering_operator.get_data().time_data.get_current_dt(),
             &grain_tracker](const unsigned int grain_id) {
              Tensor<1, dim, Number> vt = dummy_velocities;

              if (grain_tracker.get_grains().at(grain_id).n_segments() == 1 &&
                  dt != 0)
                for (unsigned int d = 0; d < dim; ++d)
                  vt[d] =
                    displacements[d][grain_id_to_eval_id.at(grain_id)] / dt;

              return vt;
            };
        }
      else
        {
          (void)mapping;

          get_velocity = [dummy_velocities = std::move(dummy_velocities),
                          &advection_mechanism,
                          &grain_tracker](const unsigned int grain_id) {
            const Tensor<1, dim, Number> vt =
              (!advection_mechanism.get_grains_data().empty() &&
               grain_tracker.get_grains().at(grain_id).n_segments() == 1) ?
                advection_mechanism.get_translation_velocity_for_grain(
                  grain_tracker.get_grain_segment_index(grain_id, segment_id)) :
                dummy_velocities;

            return vt;
          };
        }

      if (has_ghost_elements == false)
        solution.zero_out_ghost_values();

      if (Utilities::MPI::this_mpi_process(comm) != 0)
        return;

      TableHandler table;

      const std::vector labels_coords{"x", "y", "z"};
      const std::vector labels_forces{"fx", "fy", "fz"};
      const std::vector labels_torques =
        (moment_s<dim, VectorizedArrayType> == 1) ?
          std::vector({"t"}) :
          std::vector({"tx", "ty", "tz"});
      const std::vector labels_velocities{"vx", "vy", "vz"};

      const auto dummy =
        create_array<1 + dim + moment_s<dim, VectorizedArrayType>>(
          std::numeric_limits<double>::quiet_NaN());

      for (const auto &[grain_id, grain] : grain_tracker.get_grains())
        {
          table.add_value("id", grain_id);
          table.add_value("measure", grain.get_measure());
          table.add_value("radius", grain.get_max_radius());
          table.add_value("max_value", grain.get_max_value());
          table.add_value("order_parameter_id", grain.get_order_parameter_id());

          for (unsigned int d = 0; d < dim; ++d)
            table.add_value(labels_coords[d],
                            grain.n_segments() == 1 ?
                              grain.get_segments()[0].get_center()[d] :
                              std::numeric_limits<double>::quiet_NaN());

          if (advection_mechanism.enabled())
            {
              const Number *data =
                (!advection_mechanism.get_grains_data().empty() &&
                 grain.n_segments() == 1) ?
                  advection_mechanism.grain_data(
                    grain_tracker.get_grain_segment_index(grain_id, 0)) :
                  dummy.data();

              // Output volume should be less than measure
              table.add_value("volume", *data++);

              // Output forces
              for (unsigned int d = 0; d < dim; ++d)
                table.add_value(labels_forces[d], *data++);

              // Output torques
              for (unsigned int d = 0; d < moment_s<dim, VectorizedArrayType>;
                   ++d)
                table.add_value(labels_torques[d], *data++);

              // Output translation velocities
              const auto vt = get_velocity(grain_id);

              for (unsigned int d = 0; d < dim; ++d)
                table.add_value(labels_velocities[d], vt[d]);
            }
        }

      // Output to file
      std::stringstream ss;
      table.write_text(ss);

      std::ofstream out_file(output);
      out_file << ss.rdbuf();
      out_file.close();
    }

    // Output translation velocity in cell-wise manner
    template <int dim, typename Number, typename VectorizedArrayType>
    void
    add_translation_velocities_vectors(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
        &advection_mechanism,
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data,
      DataOut<dim> &                                         data_out,
      const std::string                                      prefix = "trans")
    {
      std::vector<Vector<double>> velocities;

      AdvectionVelocityData<dim, Number, VectorizedArrayType> advection_data(
        advection_mechanism, sintering_data);

      const unsigned int n_order_parameters = sintering_data.n_grains();

      for (unsigned int ig = 0; ig < n_order_parameters; ++ig)
        for (unsigned int d = 0; d < dim; ++d)
          velocities.emplace_back(
            matrix_free.get_dof_handler().get_triangulation().n_active_cells());

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          advection_data.reinit(cell);

          for (unsigned int ig = 0; ig < n_order_parameters; ++ig)
            {
              if (advection_data.has_velocity(ig))
                {
                  const auto vt = advection_data.get_translation_velocity(ig);

                  for (unsigned int ilane = 0;
                       ilane <
                       matrix_free.n_active_entries_per_cell_batch(cell);
                       ++ilane)
                    {
                      const auto icell =
                        matrix_free.get_cell_iterator(cell, ilane);

                      for (unsigned int d = 0; d < dim; ++d)
                        velocities[dim * ig + d][icell->active_cell_index()] =
                          vt[d][ilane];
                    }
                }
            }
        }

      for (unsigned int ig = 0; ig < n_order_parameters; ++ig)
        for (unsigned int d = 0; d < dim; ++d)
          data_out.add_data_vector(velocities[dim * ig + d],
                                   prefix + std::to_string(ig));
    }

    // Compute average coordination number of the packing
    template <int dim, typename Number>
    double
    compute_average_coordination_number(
      const DoFHandler<dim> &                       dof_handler,
      const unsigned int                            n_op,
      const GrainTracker::Tracker<dim, Number> &    grain_tracker,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter = nullptr)
    {
      const auto &grains = grain_tracker.get_grains();

      std::map<unsigned int, std::set<unsigned int>> neighbors;

      for (auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          const auto cell_index = cell->global_active_cell_index();

          std::vector<unsigned int> grains_at_cell;

          for (unsigned int op = 0; op < n_op; ++op)
            {
              const auto particle_id_for_op =
                grain_tracker.get_particle_index(op, cell_index);

              if (particle_id_for_op != numbers::invalid_unsigned_int)
                {
                  const auto grain_id =
                    grain_tracker.get_grain_and_segment(op, particle_id_for_op)
                      .first;

                  if (grain_id != numbers::invalid_unsigned_int)
                    grains_at_cell.push_back(grain_id);
                }
            }

          for (unsigned int i = 0; i < grains_at_cell.size(); ++i)
            {
              // Check if the grain segments are fully inside the box
              if (box_filter)
                {
                  const auto &grain = grains.at(grains_at_cell[i]);

                  bool grain_inside = true;

                  for (const auto &segment : grain.get_segments())
                    {
                      auto bottom_left = segment.get_center();
                      auto top_right   = segment.get_center();
                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          bottom_left[d] -= segment.get_radius();
                          top_right[d] += segment.get_radius();
                        }

                      // Skip the grain if either point is outside
                      if (box_filter->point_outside(bottom_left) ||
                          box_filter->point_outside(top_right))
                        {
                          grain_inside = false;
                          break;
                        }
                    }

                  if (!grain_inside)
                    continue;
                }

              for (unsigned int j = 0; j < grains_at_cell.size(); ++j)
                if (i != j)
                  neighbors[grains_at_cell[i]].insert(grains_at_cell[j]);
            }
        }

      // Perform global communication, we need to repack data since std::set can
      // not be serialized with boost
      std::vector<unsigned int> neighbors_flatten;
      for (const auto &[grain_id, grain_neighbors] : neighbors)
        {
          neighbors_flatten.push_back(grain_id);
          neighbors_flatten.push_back(grain_neighbors.size());
          std::copy(grain_neighbors.begin(),
                    grain_neighbors.end(),
                    std::back_inserter(neighbors_flatten));
        }

      auto all_neighbors =
        Utilities::MPI::all_gather(dof_handler.get_communicator(),
                                   neighbors_flatten);

      for (const auto &local_neighbors : all_neighbors)
        for (auto it = local_neighbors.begin(); it != local_neighbors.end();)
          {
            const auto grain_id    = *it++;
            const auto n_neighbors = *it++;

            neighbors[grain_id];

            auto it_begin = it;
            std::advance(it, n_neighbors);
            std::copy(it_begin,
                      it,
                      std::inserter(neighbors.at(grain_id),
                                    neighbors.at(grain_id).end()));
          }

      double avg_coord_num = 0;
      if (neighbors.size())
        avg_coord_num = std::accumulate(neighbors.begin(),
                                        neighbors.end(),
                                        0.,
                                        [](const auto &a, const auto &b) {
                                          return a + b.second.size();
                                        }) /
                        neighbors.size();

      return avg_coord_num;
    }

  } // namespace Postprocessors
} // namespace Sintering