#pragma once

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>
#include <pf-applications/grain_tracker/tracker.h>

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
    namespace internal
    {
      template <int dim, typename VectorType>
      bool
      coarsen_triangulation(
        parallel::distributed::Triangulation<dim> &tria_copy,
        const DoFHandler<dim> &                    background_dof_handler,
        DoFHandler<dim> &  background_dof_handler_coarsened,
        const VectorType & vector,
        VectorType &       vector_coarsened,
        const unsigned int n_coarsening_steps)
      {
        if (n_coarsening_steps == 0)
          return false;

        tria_copy.copy_triangulation(
          background_dof_handler.get_triangulation());
        background_dof_handler_coarsened.reinit(tria_copy);
        background_dof_handler_coarsened.distribute_dofs(
          background_dof_handler.get_fe_collection());

        // 1) copy solution so that it has the right ghosting
        const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          background_dof_handler_coarsened.locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(
            background_dof_handler_coarsened),
          background_dof_handler_coarsened.get_communicator());

        vector_coarsened.reinit(vector.n_blocks());

        for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
          {
            vector_coarsened.block(b).reinit(partitioner);
            vector_coarsened.block(b).copy_locally_owned_data_from(
              vector.block(b));
          }

        vector_coarsened.update_ghost_values();

        for (unsigned int i = 0; i < n_coarsening_steps; ++i)
          {
            // 2) mark cells for refinement
            for (const auto &cell : tria_copy.active_cell_iterators())
              if (cell->is_locally_owned() &&
                  (static_cast<unsigned int>(cell->level() + 1) ==
                   tria_copy.n_global_levels()))
                cell->set_coarsen_flag();

            // 3) perform interpolation and initialize data structures
            tria_copy.prepare_coarsening_and_refinement();

            parallel::distributed::
              SolutionTransfer<dim, typename VectorType::BlockType>
                solution_trans(background_dof_handler_coarsened);

            std::vector<const typename VectorType::BlockType *>
              vector_coarsened_ptr(vector_coarsened.n_blocks());
            for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
              vector_coarsened_ptr[b] = &vector_coarsened.block(b);

            solution_trans.prepare_for_coarsening_and_refinement(
              vector_coarsened_ptr);

            tria_copy.execute_coarsening_and_refinement();

            background_dof_handler_coarsened.distribute_dofs(
              background_dof_handler.get_fe_collection());

            const auto partitioner =
              std::make_shared<Utilities::MPI::Partitioner>(
                background_dof_handler_coarsened.locally_owned_dofs(),
                DoFTools::extract_locally_relevant_dofs(
                  background_dof_handler_coarsened),
                background_dof_handler_coarsened.get_communicator());

            for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
              vector_coarsened.block(b).reinit(partitioner);

            std::vector<typename VectorType::BlockType *> solution_ptr(
              vector_coarsened.n_blocks());
            for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
              solution_ptr[b] = &vector_coarsened.block(b);

            solution_trans.interpolate(solution_ptr);
            vector_coarsened.update_ghost_values();
          }

        return true;
      }

      template <int dim, typename VectorType>
      void
      filter_mesh_withing_bounding_box(
        const Mapping<dim> &                   mapping,
        const DoFHandler<dim> &                background_dof_handler,
        VectorType &                           vector,
        const double                           iso_level,
        std::function<int(const Point<dim> &)> box_filter,
        const double                           null_value = 0.)
      {
        AssertThrow(std::abs(iso_level - null_value) >
                      std::numeric_limits<double>::epsilon(),
                    ExcMessage(
                      "iso_level = " + std::to_string(iso_level) +
                      " and null_value = " + std::to_string(null_value) +
                      " have to be different"));

        const auto &  fe = background_dof_handler.get_fe();
        FEValues<dim> fe_values(mapping,
                                fe,
                                fe.get_unit_support_points(),
                                update_quadrature_points);

        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

        for (const auto &cell : background_dof_handler.active_cell_iterators())
          {
            if (!cell->is_locally_owned())
              continue;

            cell->get_dof_indices(dof_indices);

            fe_values.reinit(cell);

            for (unsigned int b = 0; b < vector.n_blocks(); ++b)
              {
                const auto &points = fe_values.get_quadrature_points();

                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                  {
                    const int pred_status = box_filter(points[i]);

                    auto &global_dof_value = vector.block(b)[dof_indices[i]];
                    if (pred_status == 0)
                      global_dof_value = std::min(global_dof_value, iso_level);
                    else if (pred_status == -1)
                      global_dof_value = null_value;
                  }
              }
          }
      }

      template <int dim, typename VectorType>
      bool
      build_grain_boundaries_mesh(
        Triangulation<dim - 1, dim> &          tria,
        const Mapping<dim> &                   mapping,
        const DoFHandler<dim> &                background_dof_handler,
        const VectorType &                     vector,
        const double                           iso_level,
        const unsigned int                     n_grains,
        const double                           gb_lim             = 0.14,
        const unsigned int                     n_coarsening_steps = 0,
        std::function<int(const Point<dim> &)> box_filter         = nullptr,
        const unsigned int                     n_subdivisions     = 1,
        const double                           tolerance          = 1e-10)
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
            internal::coarsen_triangulation(tria_copy,
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

            internal::filter_mesh_withing_bounding_box(
              mapping,
              *background_dof_handler_to_be_used,
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
      const GrainTracker::Tracker<dim, Number> &grain_tracker_in,
      const unsigned int                        n_subdivisions = 1,
      const double                              tolerance      = 1e-10)
    {
      (void)mapping;


      const auto comm = background_dof_handler.get_communicator();

      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

      auto grain_tracker = grain_tracker_in.clone();
      grain_tracker->track(vector, n_op, true);

      std::ofstream outfile(filename);

      const auto grains = grain_tracker->get_grains();

      unsigned int n_grains = 0;

      for (const auto &entry : grains)
        n_grains = std::max(entry.first, n_grains);
      n_grains++;

      const auto bb = GridTools::compute_bounding_box(
        background_dof_handler.get_triangulation());

      std::vector<Number> parameters((dim + 1) * n_grains, 0);

      for (unsigned int b = 0; b < n_op; ++b)
        {
          for (const auto &cell :
               background_dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                if (grain_tracker->get_particle_index(
                      b, cell->global_active_cell_index()) ==
                    numbers::invalid_unsigned_int)
                  continue;

                const auto grain_id =
                  grain_tracker
                    ->get_grain_and_segment(b,
                                            grain_tracker->get_particle_index(
                                              b,
                                              cell->global_active_cell_index()))
                    .first;

                if (grain_id == numbers::invalid_unsigned_int)
                  continue;

                const auto center  = cell->center();
                const auto measure = cell->measure();

                for (unsigned int d = 0; d < dim; ++d)
                  parameters[grain_id * (dim + 1) + d] += center[d] * measure;

                parameters[grain_id * (dim + 1) + dim] += measure;
              }
        }

      Utilities::MPI::sum(parameters, comm, parameters);

      for (unsigned int g = 0; g < n_grains; ++g)
        {
          for (unsigned int d = 0; d < dim; ++d)
            if (parameters[g * (dim + 1) + dim] != 0.0)
              parameters[g * (dim + 1) + d] /= parameters[g * (dim + 1) + dim];
          parameters[g * (dim + 1) + dim] =
            std::sqrt(parameters[g * (dim + 1) + dim] / numbers::PI);
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
                if (grain_tracker->get_particle_index(
                      b, cell->global_active_cell_index()) ==
                    numbers::invalid_unsigned_int)
                  continue;

                const auto grain_id =
                  grain_tracker
                    ->get_grain_and_segment(b,
                                            grain_tracker->get_particle_index(
                                              b,
                                              cell->global_active_cell_index()))
                    .first;

                if (grain_id == numbers::invalid_unsigned_int)
                  continue;

                mc.process_cell(cell,
                                vector.block(b + 2),
                                iso_level,
                                points_local[grain_id]);
              }
        }


      const auto points_global =
        Utilities::MPI::reduce<std::vector<std::vector<Point<dim>>>>(
          points_local, comm, [](const auto &a, const auto &b) {
            std::vector<std::vector<Point<dim>>> result = a;

            for (unsigned int i = 0; i < a.size(); ++i)
              {
                result[i].insert(result[i].end(), b[i].begin(), b[i].end());
              }

            return result;
          });

      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          std::ofstream outfile(filename);

          outfile << n_grains << std::endl;
          outfile << n_op << std::endl;
          for (const auto &entry : grains)
            outfile << entry.second.get_order_parameter_id() << " ";
          outfile << std::endl;

          for (unsigned int d = 0; d < dim; ++d)
            outfile << bb.get_boundary_points().first[d] << " ";
          outfile << std::endl;

          for (unsigned int d = 0; d < dim; ++d)
            outfile << bb.get_boundary_points().second[d] << " ";
          outfile << std::endl;

          for (const auto &i : parameters)
            outfile << i << " ";
          outfile << std::endl;

          for (const auto &points : points_global)
            {
              for (const auto &point : points)
                outfile << point << " ";
              outfile << std::endl;
            }
        }

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();
    }

    template <int dim, typename VectorType, typename Number>
    void
    output_grain_contours_vtu(
      const Mapping<dim> &                      mapping,
      const DoFHandler<dim> &                   background_dof_handler,
      const VectorType &                        vector,
      const double                              iso_level,
      const std::string                         filename,
      const unsigned int                        n_grains,
      const GrainTracker::Tracker<dim, Number> &grain_tracker_in,
      const unsigned int                        n_coarsening_steps = 0,
      std::function<int(const Point<dim> &)>    box_filter         = nullptr,
      const unsigned int                        n_subdivisions     = 1,
      const double                              tolerance          = 1e-10)
    {
      std::shared_ptr<GrainTracker::Tracker<dim, Number>> grain_tracker;
      if (n_coarsening_steps == 0)
        grain_tracker = grain_tracker_in.clone();

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
          internal::coarsen_triangulation(tria_copy,
                                          background_dof_handler,
                                          dof_handler_copy,
                                          vector,
                                          solution_dealii,
                                          n_coarsening_steps);

          vector_to_be_used                 = &solution_dealii;
          background_dof_handler_to_be_used = &dof_handler_copy;
        }

      if (grain_tracker)
        {
          if (grain_tracker->get_grains().empty())
            grain_tracker->initial_setup(vector, n_grains);
          else
            grain_tracker->track(vector, n_grains, true);
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

          internal::filter_mesh_withing_bounding_box(
            mapping,
            *background_dof_handler_to_be_used,
            *only_order_params,
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

      for (unsigned int b = 0; b < n_grains; ++b)
        {
          for (const auto &cell :
               background_dof_handler_to_be_used->active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const unsigned int old_size = cells.size();

                mc.process_cell(cell,
                                vector_to_be_used->block(b + 2),
                                iso_level,
                                vertices,
                                cells);

                for (unsigned int i = old_size; i < cells.size(); ++i)
                  {
                    if (grain_tracker)
                      cells[i].material_id =
                        grain_tracker
                          ->get_grain_and_segment(
                            b,
                            grain_tracker->get_particle_index(
                              b, cell->global_active_cell_index()))
                          .first;

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
              vector_grain_id[cell->active_cell_index()] = cell->material_id();
              vector_order_parameter_id[cell->active_cell_index()] =
                cell->manifold_id();
            }
        }
      else
        {
          vector_grain_id           = -1.0; // initialized with dummy value
          vector_order_parameter_id = -1.0;
        }

      Vector<float> vector_rank(tria.n_active_cells());
      vector_rank = Utilities::MPI::this_mpi_process(
        background_dof_handler.get_communicator());

      // step 2) output mesh
      MyDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(vector_grain_id, "grain_id");
      data_out.add_data_vector(vector_order_parameter_id, "order_parameter_id");
      data_out.add_data_vector(vector_rank, "subdomain");

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();
    }

    template <int dim, typename VectorType>
    void
    output_grain_boundaries_vtu(
      const Mapping<dim> &                   mapping,
      const DoFHandler<dim> &                background_dof_handler,
      const VectorType &                     vector,
      const double                           iso_level,
      const std::string                      filename,
      const unsigned int                     n_grains,
      const double                           gb_lim             = 0.14,
      const unsigned int                     n_coarsening_steps = 0,
      std::function<int(const Point<dim> &)> box_filter         = nullptr,
      const unsigned int                     n_subdivisions     = 1,
      const double                           tolerance          = 1e-10)
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
      MyDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());
    }

    template <int dim, typename VectorType>
    void
    output_concentration_contour_vtu(
      const Mapping<dim> &                   mapping,
      const DoFHandler<dim> &                background_dof_handler,
      const VectorType &                     vector,
      const double                           iso_level,
      const std::string                      filename,
      const unsigned int                     n_coarsening_steps = 0,
      std::function<int(const Point<dim> &)> box_filter         = nullptr,
      const unsigned int                     n_subdivisions     = 1,
      const double                           tolerance          = 1e-10)
    {
      const auto  only_concentration_ptr = vector.create_view(0, 1);
      const auto &only_concentration     = *only_concentration_ptr;

      const bool has_ghost_elements = only_concentration.has_ghost_elements();

      if (has_ghost_elements == false)
        only_concentration.update_ghost_values();

      auto vector_to_be_used                 = &only_concentration;
      auto background_dof_handler_to_be_used = &background_dof_handler;

      parallel::distributed::Triangulation<dim> tria_copy(
        background_dof_handler.get_communicator());
      DoFHandler<dim> dof_handler_copy;
      VectorType      solution_dealii;

      if (n_coarsening_steps != 0)
        {
          internal::coarsen_triangulation(tria_copy,
                                          background_dof_handler,
                                          dof_handler_copy,
                                          only_concentration,
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
              solution_dealii = only_concentration;
              solution_dealii.update_ghost_values();
              vector_to_be_used = &solution_dealii;
            }

          internal::filter_mesh_withing_bounding_box(
            mapping,
            *background_dof_handler_to_be_used,
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
      MyDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(vector_rank, "subdomain");

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());

      if (has_ghost_elements == false)
        only_concentration.zero_out_ghost_values();
    }

    template <int dim, typename VectorType>
    typename VectorType::value_type
    compute_surface_area(
      const Mapping<dim> &                    mapping,
      const DoFHandler<dim> &                 background_dof_handler,
      const VectorType &                      vector,
      const double                            iso_level,
      std::function<bool(const Point<dim> &)> predicate      = nullptr,
      const unsigned int                      n_subdivisions = 1,
      const double                            tolerance      = 1e-10)
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
                (!predicate || predicate(cell->center())))
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
      const Mapping<dim> &                    mapping,
      const DoFHandler<dim> &                 background_dof_handler,
      const VectorType &                      vector,
      const double                            iso_level,
      const unsigned int                      n_grains,
      const double                            gb_lim         = 0.14,
      std::function<bool(const Point<dim> &)> predicate      = nullptr,
      const unsigned int                      n_subdivisions = 1,
      const double                            tolerance      = 1e-10)
    {
      Triangulation<dim - 1, dim> tria;

      const unsigned int                     n_coarsening_steps = 0;
      std::function<int(const Point<dim> &)> box_filter         = nullptr;

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

      typename VectorType::value_type gb_area = 0;
      if (tria_not_empty)
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned() &&
              (!predicate || predicate(cell->center())))
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
      run_flooding(const typename DoFHandler<dim>::cell_iterator &cell,
                   const BlockVectorType &                        solution,
                   LinearAlgebra::distributed::Vector<Number> &   particle_ids,
                   const unsigned int                             id)
      {
        const double threshold_lower     = 0.8;  // TODO
        const double invalid_particle_id = -1.0; // TODO

        if (cell->has_children())
          {
            unsigned int counter = 0;

            for (const auto &child : cell->child_iterators())
              counter += run_flooding<dim>(child, solution, particle_ids, id);

            return counter;
          }

        if (cell->is_locally_owned() == false)
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

                if (values.linfty_norm() >= threshold_lower)
                  return 0;
              }
          }
        else
          {
            cell->get_dof_values(solution.block(0), values);

            if (values.linfty_norm() >= threshold_lower)
              return 0;
          }

        particle_ids[cell->global_active_cell_index()] = id;

        unsigned int counter = 1;

        for (const auto face : cell->face_indices())
          if (cell->at_boundary(face) == false)
            counter += run_flooding<dim>(cell->neighbor(face),
                                         solution,
                                         particle_ids,
                                         id);

        return counter;
      }
    } // namespace internal

    template <int dim, typename VectorType>
    void
    estimate_porosity(const Mapping<dim> &   mapping,
                      const DoFHandler<dim> &dof_handler,
                      const VectorType &     solution,
                      const std::string      output)
    {
      const double invalid_particle_id = -1.0; // TODO

      const auto tria = dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &dof_handler.get_triangulation());

      AssertThrow(tria, ExcNotImplemented());

      const auto comm = dof_handler.get_communicator();

      LinearAlgebra::distributed::Vector<double> particle_ids(
        tria->global_active_cell_index_partitioner().lock());

      // step 1) run flooding and determine local particles and give them
      // local ids
      particle_ids = invalid_particle_id;

      unsigned int counter = 0;
      unsigned int offset  = 0;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (internal::run_flooding<dim>(cell, solution, particle_ids, counter) >
            0)
          counter++;

      // step 2) determine the global number of locally determined particles
      // and give each one an unique id by shifting the ids
      MPI_Exscan(&counter, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

      for (auto &particle_id : particle_ids)
        if (particle_id != invalid_particle_id)
          particle_id += offset;

      // step 3) get particle ids on ghost cells and figure out if local
      // particles and ghost particles might be one particle
      particle_ids.update_ghost_values();

      std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
        local_connectiviy(counter);

      for (const auto &ghost_cell :
           dof_handler.get_triangulation().active_cell_iterators())
        if (ghost_cell->is_ghost())
          {
            const auto particle_id =
              particle_ids[ghost_cell->global_active_cell_index()];

            if (particle_id == invalid_particle_id)
              continue;

            for (const auto face : ghost_cell->face_indices())
              {
                if (ghost_cell->at_boundary(face))
                  continue;

                const auto add = [&](const auto &ghost_cell,
                                     const auto &local_cell) {
                  if (local_cell->is_locally_owned() == false)
                    return;

                  const auto neighbor_particle_id =
                    particle_ids[local_cell->global_active_cell_index()];

                  if (neighbor_particle_id == invalid_particle_id)
                    return;

                  auto &temp = local_connectiviy[neighbor_particle_id - offset];
                  temp.emplace_back(ghost_cell->subdomain_id(), particle_id);
                  std::sort(temp.begin(), temp.end());
                  temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
                };

                if (ghost_cell->neighbor(face)->has_children())
                  {
                    for (unsigned int subface = 0;
                         subface <
                         GeometryInfo<dim>::n_subfaces(
                           dealii::internal::SubfaceCase<dim>::case_isotropic);
                         ++subface)
                      add(ghost_cell,
                          ghost_cell->neighbor_child_on_subface(face, subface));
                  }
                else
                  add(ghost_cell, ghost_cell->neighbor(face));
              }
          }

      // step 4) based on the local-ghost information, figure out all
      // particles on all processes that belong togher (unification ->
      // clique), give each clique an unique id, and return mapping from the
      // global non-unique ids to the global ids
      const auto local_to_global_particle_ids =
        GrainTracker::perform_distributed_stitching(comm, local_connectiviy);

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

      const auto next_cell = [&](const auto &, const auto cell_in) {
        auto cell = cell_in;
        cell++;

        while (cell != tria->end())
          {
            if (cell->is_active() && cell->is_locally_owned() &&
                cell_to_id[cell->active_cell_index()] != invalid_particle_id)
              break;

            ++cell;
          }

        return cell;
      };

      const auto first_cell = [&](const auto &tria) {
        return next_cell(tria, tria.begin());
      };

      data_out.set_cell_selection(first_cell, next_cell);

      data_out.attach_triangulation(dof_handler.get_triangulation());
      data_out.add_data_vector(cell_to_id, "ids");
      data_out.build_patches(mapping);
      data_out.write_vtu_in_parallel(output, dof_handler.get_communicator());
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
                       const VectorType &     solution)
    {
      const double threshold = 0.5 - 1e-2;
      const double abs_tol   = 1e-15;

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
              const double c_norm = values.linfty_norm();

              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (min_cells[d].first.state() == IteratorState::invalid ||
                      (std::abs(cell->barycenter()[d] -
                                min_cells[d].first->barycenter()[d]) <
                         abs_tol &&
                       c_norm > min_cells[d].second) ||
                      cell->barycenter()[d] <
                        min_cells[d].first->barycenter()[d])
                    {
                      min_cells[d].first  = cell;
                      min_cells[d].second = c_norm;
                    }

                  if (max_cells[d].first.state() == IteratorState::invalid ||
                      (std::abs(cell->barycenter()[d] -
                                max_cells[d].first->barycenter()[d]) <
                         abs_tol &&
                       c_norm > max_cells[d].second) ||
                      cell->barycenter()[d] >
                        max_cells[d].first->barycenter()[d])
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
      const unsigned int            n_intervals = 10;
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
                       const std::string      output)
    {
      const auto bb = estimate_shrinkage(mapping, dof_handler, solution);

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

        solution.zero_out_ghost_values();
      }
    } // namespace internal

    /* Estimate mesh quality: 0 - low, 1 - high */
    template <int dim, typename BlockVectorType>
    void
    estimate_mesh_quality(const Mapping<dim> &   mapping,
                          const DoFHandler<dim> &dof_handler,
                          const BlockVectorType &solution,
                          const std::string      output)
    {
      Vector<typename BlockVectorType::value_type> quality(
        dof_handler.get_triangulation().n_active_cells());

      auto callback =
        [&quality](const typename BlockVectorType::value_type qval,
                   const DoFCellAccessor<dim, dim, false> &   cell) {
          quality[cell.active_cell_index()] = qval;
        };

      internal::do_estimate_mesh_quality<dim>(dof_handler, solution, callback);

      DataOut<dim> data_out;
      data_out.attach_triangulation(dof_handler.get_triangulation());
      data_out.add_data_vector(quality, "quality");
      data_out.build_patches(mapping);
      data_out.write_vtu_in_parallel(output, dof_handler.get_communicator());
    }

    /* Estimate min mesh quality: 0 - low, 1 - high */
    template <int dim, typename BlockVectorType>
    typename BlockVectorType::value_type
    estimate_mesh_quality_min(const DoFHandler<dim> &dof_handler,
                              const BlockVectorType &solution)
    {
      typename BlockVectorType::value_type quality = 1.;

      auto callback =
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

      std::vector<QuantityCallback> quantities;

      for (const auto &qty : labels)
        {
          QuantityCallback callback;

          if (qty == "solid_vol")
            callback = [](const VectorizedArrayType *                value,
                          const Tensor<1, dim, VectorizedArrayType> *gradient,
                          const unsigned int                         n_grains) {
              (void)gradient;
              (void)n_grains;

              return value[0];
            };
          else if (qty == "surf_area")
            callback = [](const VectorizedArrayType *                value,
                          const Tensor<1, dim, VectorizedArrayType> *gradient,
                          const unsigned int                         n_grains) {
              (void)gradient;
              (void)n_grains;

              return value[0] * (1.0 - value[0]);
            };
          else if (qty == "gb_area")
            callback = [](const VectorizedArrayType *                value,
                          const Tensor<1, dim, VectorizedArrayType> *gradient,
                          const unsigned int                         n_grains) {
              (void)gradient;

              VectorizedArrayType eta_ij_sum = 0.0;
              for (unsigned int i = 0; i < n_grains; ++i)
                for (unsigned int j = i + 1; j < n_grains; ++j)
                  eta_ij_sum += value[2 + i] * value[2 + j];

              return eta_ij_sum;
            };
          else if (qty == "avg_grain_size")
            callback = [](const VectorizedArrayType *                value,
                          const Tensor<1, dim, VectorizedArrayType> *gradient,
                          const unsigned int                         n_grains) {
              (void)gradient;

              VectorizedArrayType eta_i2_sum = 0.0;
              for (unsigned int i = 0; i < n_grains; ++i)
                eta_i2_sum += value[2 + i] * value[2 + i];

              return eta_i2_sum;
            };
          else if (qty == "surf_area_nrm")
            callback = [](const VectorizedArrayType *                value,
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
            };
          else if (qty == "free_energy")
            callback = [&sintering_data](
                         const VectorizedArrayType *                value,
                         const Tensor<1, dim, VectorizedArrayType> *gradient,
                         const unsigned int                         n_grains) {
              (void)gradient;

              const VectorizedArrayType &c = value[0];

              std::vector<VectorizedArrayType> etas(n_grains);
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                etas[ig] = value[2 + ig];

              return sintering_data.free_energy.f(c, etas);
            };
          else
            AssertThrow(false,
                        ExcMessage("Invalid domain integral provided: " + qty));

          quantities.push_back(callback);
        }

      return quantities;
    }

  } // namespace Postprocessors
} // namespace Sintering