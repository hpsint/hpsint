#pragma once

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

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
    template <int dim, typename VectorType>
    void
    output_grain_contours(const Mapping<dim> &   mapping,
                          const DoFHandler<dim> &background_dof_handler,
                          const VectorType &     vector,
                          const double           iso_level,
                          const std::string      filename,
                          const unsigned int     n_coarsening_steps = 0,
                          const unsigned int     n_subdivisions     = 1,
                          const double           tolerance          = 1e-10)
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
              dof_handler_copy.get_communicator());

          solution_dealii.reinit(vector.n_blocks());

          for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
            {
              solution_dealii.block(b).reinit(partitioner);
              solution_dealii.block(b).copy_locally_owned_data_from(
                vector.block(b));
            }

          solution_dealii.update_ghost_values();

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
                  dof_handler_copy.get_communicator());

              for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
                solution_dealii.block(b).reinit(partitioner);

              std::vector<typename VectorType::BlockType *> solution_ptr(
                solution_dealii.n_blocks());
              for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
                solution_ptr[b] = &solution_dealii.block(b);

              solution_trans.interpolate(solution_ptr);
              solution_dealii.update_ghost_values();
            }

          vector_to_be_used                 = &solution_dealii;
          background_dof_handler_to_be_used = &dof_handler_copy;
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

      for (unsigned int b = 0; b < vector_to_be_used->n_blocks() - 2; ++b)
        {
          const unsigned int old_size = cells.size();

          mc.process(*background_dof_handler_to_be_used,
                     vector_to_be_used->block(b + 2),
                     iso_level,
                     vertices,
                     cells);

          for (unsigned int i = old_size; i < cells.size(); ++i)
            cells[i].material_id = b;
        }

      Triangulation<dim - 1, dim> tria;

      if (vertices.size() > 0)
        tria.create_triangulation(vertices, cells, subcelldata);
      else
        GridGenerator::hyper_cube(tria, -1e-6, 1e-6);

      Vector<float> vector_grain_id(tria.n_active_cells());
      for (const auto cell : tria.active_cell_iterators())
        vector_grain_id[cell->active_cell_index()] = cell->material_id();

      Vector<float> vector_rank(tria.n_active_cells());
      vector_rank = Utilities::MPI::this_mpi_process(
        background_dof_handler.get_communicator());

      // step 2) output mesh
      MyDataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(vector_grain_id, "grain_id");
      data_out.add_data_vector(vector_rank, "subdomain");

      data_out.build_patches();
      data_out.write_vtu_in_parallel(filename,
                                     background_dof_handler.get_communicator());

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();
    }



    template <int dim, typename VectorType>
    void
    estimate_overhead(const Mapping<dim> &   mapping,
                      const DoFHandler<dim> &background_dof_handler,
                      const VectorType &     vector,
                      const bool             output_mesh = false)
    {
      using Number = typename VectorType::value_type;

      const unsigned int n_active_cells_0 =
        background_dof_handler.get_triangulation().n_global_active_cells();
      unsigned int n_active_cells_1 = 0;

      if (output_mesh)
        {
          DataOut<dim> data_out;
          data_out.attach_triangulation(
            background_dof_handler.get_triangulation());
          data_out.build_patches(mapping);
          data_out.write_vtu_in_parallel(
            "reduced_mesh.0.vtu", background_dof_handler.get_communicator());
        }

      for (unsigned int b = 0; b < vector.n_blocks() - 2; ++b)
        {
          parallel::distributed::Triangulation<dim> tria_copy(
            background_dof_handler.get_communicator());
          DoFHandler<dim> dof_handler_copy;
          VectorType      solution_dealii;

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
              dof_handler_copy.get_communicator());

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
                  dof_handler_copy.get_communicator());

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
              data_out.write_vtu_in_parallel(
                "reduced_mesh." + std::to_string(b + 1) + ".vtu",
                background_dof_handler.get_communicator());
            }
        }

      ConditionalOStream pcout(std::cout,
                               Utilities::MPI::this_mpi_process(
                                 background_dof_handler.get_communicator()) ==
                                 0);

      pcout << "Estimation of mesh overhead : "
            << std::to_string((n_active_cells_0 * (vector.n_blocks() - 2)) *
                                100 / n_active_cells_1 -
                              100)
            << "%" << std::endl
            << std::endl;
    }

  } // namespace Postprocessors
} // namespace Sintering