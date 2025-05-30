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

namespace dealii
{
  namespace parallel
  {
    namespace distributed
    {
      template <int dim, typename VectorType, int spacedim = dim>
      class SolutionSerialization
      {
      public:
        using value_type = typename VectorType::value_type;
        using size_type  = types::global_dof_index;

        SolutionSerialization(const DoFHandler<dim, spacedim> &dof_handler)
          : comm(dof_handler.get_communicator())
          , n_locally_owned_dofs(dof_handler.n_locally_owned_dofs())
          , unique_dof_map(create_unique_dof_map(dof_handler))
        {}

        void
        add_vectors(std::vector<VectorType *> vectors)
        {
          for (auto v : vectors)
            this->vectors.emplace_back(v);
        }

        void
        add_vectors(std::vector<const VectorType *> vectors)
        {
          for (auto v : vectors)
            this->vectors.emplace_back(const_cast<VectorType *>(v));
        }

        void
        save(const std::string file_name)
        {
          // determine local size
          const size_type local_size = n_locally_owned_dofs * vectors.size();

          // collect all values in a single vector
          std::vector<value_type> temp(local_size);

          for (unsigned int b = 0, c = 0; b < vectors.size(); ++b)
            for (unsigned int i = 0; i < n_locally_owned_dofs; ++i, ++c)
              temp[c] = vectors[b]->local_element(unique_dof_map[i]);

          // write to hard drive
          mpi_io_read_or_write(false /*write*/, file_name, temp);
        }

        void
        load(const std::string file_name)
        {
          // determine local size
          const size_type local_size = n_locally_owned_dofs * vectors.size();

          // read from hard drive
          std::vector<value_type> temp(local_size);
          mpi_io_read_or_write(true /*read*/, file_name, temp);

          // split up single vector into blocks
          for (unsigned int b = 0, c = 0; b < vectors.size(); ++b)
            for (unsigned int i = 0; i < n_locally_owned_dofs; ++i, ++c)
              vectors[b]->local_element(unique_dof_map[i]) = temp[c];
        }

      private:
        const MPI_Comm                  comm;
        const size_type                 n_locally_owned_dofs;
        const std::vector<unsigned int> unique_dof_map;
        std::vector<VectorType *>       vectors;

        static void
        visit_cells_recursevely(
          const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
          const IndexSet            &locally_owned_dofs,
          std::vector<bool>         &mask,
          std::vector<unsigned int> &result)
        {
          if (cell->is_active())
            {
              if (cell->is_locally_owned())
                {
                  std::vector<types::global_dof_index> dof_indices(
                    cell->get_fe().n_dofs_per_cell());

                  cell->get_dof_indices(dof_indices);

                  for (const auto i : dof_indices)
                    {
                      if (locally_owned_dofs.is_element(i) == false)
                        continue;

                      const auto i_local =
                        locally_owned_dofs.index_within_set(i);

                      if (mask[i_local] == true)
                        continue;

                      mask[i_local] = true;
                      result.push_back(i_local);
                    }
                }
            }
          else
            {
              for (const auto &child : cell->child_iterators())
                visit_cells_recursevely(child,
                                        locally_owned_dofs,
                                        mask,
                                        result);
            }
        }

        std::vector<unsigned int> static create_unique_dof_map(
          const DoFHandler<dim, spacedim> &dof_handler)
        {
          const auto &locally_owned_dofs = dof_handler.locally_owned_dofs();

          std::vector<unsigned int> result;
          result.reserve(locally_owned_dofs.n_elements());

          std::vector<bool> mask(locally_owned_dofs.n_elements(), false);

          for (const auto &cell : dof_handler.cell_iterators_on_level(0))
            visit_cells_recursevely(cell, locally_owned_dofs, mask, result);

          AssertThrow(result.size() == locally_owned_dofs.n_elements(),
                      ExcNotImplemented());

          return result;
        }

        void
        mpi_io_read_or_write(const bool               do_read,
                             const std::string       &filename,
                             std::vector<value_type> &src) const
        {
          const size_type local_size = src.size();

          size_type offset = 0;

          int ierr = MPI_Exscan(&local_size,
                                &offset,
                                1,
                                Utilities::MPI::mpi_type_id_for_type<size_type>,
                                MPI_SUM,
                                comm);
          AssertThrowMPI(ierr);

          // local displacement in file (in bytes)
          MPI_Offset disp =
            static_cast<unsigned long int>(offset) * sizeof(value_type);

          // ooen file ...
          MPI_File fh;
          ierr = MPI_File_open(comm,
                               filename.c_str(),
                               do_read ? (MPI_MODE_RDONLY) :
                                         (MPI_MODE_CREATE | MPI_MODE_WRONLY),
                               MPI_INFO_NULL,
                               &fh);
          AssertThrowMPI(ierr);

          // ... set view
          MPI_File_set_view(fh,
                            disp,
                            Utilities::MPI::mpi_type_id_for_type<value_type>,
                            Utilities::MPI::mpi_type_id_for_type<value_type>,
                            "native",
                            MPI_INFO_NULL);

          if (do_read)
            // ... read file
            ierr = MPI_File_read_all(
              fh,
              src.data(),
              src.size(),
              Utilities::MPI::mpi_type_id_for_type<value_type>,
              MPI_STATUSES_IGNORE);
          else
            // ... write file
            ierr = MPI_File_write_all(
              fh,
              src.data(),
              src.size(),
              Utilities::MPI::mpi_type_id_for_type<value_type>,
              MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);

          // ... close file
          ierr = MPI_File_close(&fh);
          AssertThrowMPI(ierr);
        }
      };
    } // namespace distributed
  }   // namespace parallel

} // namespace dealii