#pragma once

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

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
                          const unsigned int     n_subdivisions = 1,
                          const double           tolerance      = 1e-10)
    {
      // step 0) coarsen background mesh 1 or 2 times to reduce memory
      // consumption

      // TODO

      // step 1) create surface mesh
      std::vector<Point<dim>>        vertices;
      std::vector<CellData<dim - 1>> cells;
      SubCellData                    subcelldata;

      const GridTools::MarchingCubeAlgorithm<dim,
                                             typename VectorType::BlockType>
        mc(mapping, background_dof_handler.get_fe(), n_subdivisions, tolerance);

      const bool has_ghost_elements = vector.has_ghost_elements();

      if (has_ghost_elements == false)
        vector.update_ghost_values();

      for (unsigned int b = 0; b < vector.n_blocks() - 2; ++b)
        {
          const unsigned int old_size = cells.size();

          mc.process(background_dof_handler,
                     vector.block(b + 2),
                     iso_level,
                     vertices,
                     cells);

          for (unsigned int i = old_size; i < cells.size(); ++i)
            cells[i].material_id = b;
        }

      if (has_ghost_elements == false)
        vector.zero_out_ghost_values();

      Triangulation<dim - 1, dim> tria;
      tria.create_triangulation(vertices, cells, subcelldata);

      Vector<float> vector_grain_id(tria.n_active_cells());
      for (const auto cell : tria.active_cell_iterators())
        vector_grain_id[cell->active_cell_index()] = cell->material_id();

      Vector<float> vector_rank(tria.n_active_cells());
      vector_rank = Utilities::MPI::this_mpi_process(
        background_dof_handler.get_communicator());

      // step 2) output mesh
      DataOut<dim - 1, dim> data_out;
      data_out.attach_triangulation(tria);
      data_out.add_data_vector(vector_grain_id, "grain_id");
      data_out.add_data_vector(vector_rank, "subdomain");

      data_out.build_patches(),
        data_out.write_vtu_in_parallel(
          filename, background_dof_handler.get_communicator());
    }
  } // namespace Postprocessors
} // namespace Sintering