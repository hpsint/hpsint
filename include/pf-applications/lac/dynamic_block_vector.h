#pragma once

#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/vector_operation.h>

#include <pf-applications/base/memory_consumption.h>

namespace dealii
{
  namespace LinearAlgebra
  {
    namespace distributed
    {
      template <typename T>
      class DynamicBlockVector
      {
      public:
        using BlockType  = Vector<T>;
        using value_type = T;
        using size_type  = types::global_dof_index;

        /**
         * Initialization.
         */
        explicit DynamicBlockVector(const unsigned int n = 0)
          : size_(0)
        {
          reinit(n);
        }

        explicit DynamicBlockVector(const DynamicBlockVector<T> &V)
          : size_(0)
        {
          *this = V;
        }

        DynamicBlockVector<T> &
        operator=(const DynamicBlockVector<T> &V)
        {
          block_counter = V.n_blocks();
          blocks.resize(n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            {
              if (blocks[b] == nullptr)
                blocks[b] = std::make_shared<BlockType>();
              block(b).reinit(V.block(b), true);
              block(b) = V.block(b);
            }

          size_ = 0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            size_ += block(b).size();

          return *this;
        }

        void
        reinit(const unsigned int n, const bool omit_zeroing_entries = false)
        {
          block_counter = n;

          const unsigned int old_blocks_size = blocks.size();

          if (n_blocks() > old_blocks_size)
            {
              blocks.resize(n_blocks());

              for (unsigned int b = old_blocks_size; b < n_blocks(); ++b)
                {
                  if (blocks[b] == nullptr)
                    blocks[b] = std::make_shared<BlockType>();

                  if (old_blocks_size != 0)
                    block(b).reinit(block(0), omit_zeroing_entries);
                }
            }

          size_ = 0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            size_ += block(b).size();
        }

        void
        reinit(const DynamicBlockVector<T> &V,
               const bool                   omit_zeroing_entries = false)
        {
          block_counter = V.n_blocks();
          blocks.resize(n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            {
              if (blocks[b] == nullptr)
                blocks[b] = std::make_shared<BlockType>();
              block(b).reinit(V.block(b), omit_zeroing_entries);
            }

          size_ = 0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            size_ += block(b).size();
        }

        types::global_dof_index
        size() const
        {
          return size_;
        }


        /**
         * Create view.
         */
        std::unique_ptr<const DynamicBlockVector<T>>
        create_view(unsigned int start, unsigned end) const
        {
          AssertIndexRange(start, end + 1);
          AssertIndexRange(end, n_blocks() + 1);

          auto view = std::make_unique<const DynamicBlockVector<T>>();

          for (unsigned int i = start; i < end; ++i)
            const_cast<DynamicBlockVector<T> *>(view.get())
              ->blocks.push_back(blocks[i]);

          const_cast<DynamicBlockVector<T> *>(view.get())->block_counter =
            view->blocks.size();

          return view;
        }

        std::unique_ptr<DynamicBlockVector<T>>
        create_view(unsigned int start, unsigned end)
        {
          AssertIndexRange(start, end + 1);
          AssertIndexRange(end, n_blocks() + 1);

          auto view = std::make_unique<DynamicBlockVector>();

          for (unsigned int i = start; i < end; ++i)
            view->blocks.push_back(blocks[i]);

          view->block_counter = view->blocks.size();

          return view;
        }

        /**
         * Blocks.
         */
        BlockType &
        block(const unsigned int i)
        {
          AssertIndexRange(i, n_blocks());
          return *blocks[i];
        }

        const BlockType &
        block(const unsigned int i) const
        {
          AssertIndexRange(i, n_blocks());
          return *blocks[i];
        }

        unsigned int
        n_blocks() const
        {
          return block_counter;
        }

        /**
         * Communication.
         */
        void
        update_ghost_values() const
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).update_ghost_values();
        }

        void
        zero_out_ghost_values() const
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).zero_out_ghost_values();
        }

        bool
        has_ghost_elements() const
        {
          Assert(n_blocks() > 0, ExcInternalError());

          for (unsigned int b = 1; b < n_blocks(); ++b)
            Assert(block(0).has_ghost_elements() ==
                     block(b).has_ghost_elements(),
                   ExcInternalError());

          return block(0).has_ghost_elements();
        }

        void
        compress(VectorOperation::values operation)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).compress(operation);
        }

        bool
        is_globally_compatible(
          const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner)
          const
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            if (block(b).get_partitioner().get() != partitioner.get())
              return false;

          return true;
        }

        /**
         * Computation.
         */
        T
        l2_norm() const
        {
          T result = 0.0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            result += std::pow(block(b).l2_norm(), 2.0);
          return std::sqrt(result);
        }

        T
        l1_norm() const
        {
          T result = 0.0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            result += block(b).l1_norm();
          return result;
        }

        T
        linfty_norm() const
        {
          T result = 0.0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            result = std::max<T>(result, block(b).linfty_norm());
          return result;
        }

        void
        add(const T a, const DynamicBlockVector<T> &V)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).add(a, V.block(b));
        }

        void
        add(const T                      v,
            const DynamicBlockVector<T> &V,
            const T                      w,
            const DynamicBlockVector<T> &W)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).add(v, V.block(b), w, W.block(b));
        }

        void
        sadd(const T s, const DynamicBlockVector<T> &V)
        {
          AssertDimension(n_blocks(), V.n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).sadd(s, V.block(b));
        }

        void
        sadd(const T s, const T a, const DynamicBlockVector<T> &V)
        {
          AssertDimension(n_blocks(), V.n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).sadd(s, a, V.block(b));
        }

        void
        scale(const DynamicBlockVector<T> &V)
        {
          AssertDimension(n_blocks(), V.n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).scale(V.block(b));
        }

        void
        operator*=(const T factor)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b) *= factor;
        }

        T
        add_and_dot(const T                      a,
                    const DynamicBlockVector<T> &V,
                    const DynamicBlockVector<T> &W)
        {
          AssertDimension(n_blocks(), V.n_blocks());
          AssertDimension(n_blocks(), W.n_blocks());

          T result = 0.0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            result += block(b).add_and_dot(a, V.block(b), W.block(b));
          return result;
        }

        void
        operator=(const T &v)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b) = v;
        }

        T
        operator*(const DynamicBlockVector<T> &V)
        {
          AssertDimension(n_blocks(), V.n_blocks());

          T result = 0.0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            result += block(b) * V.block(b);
          return result;
        }

        void
        copy_locally_owned_data_from(const DynamicBlockVector<T> &V)
        {
          AssertDimension(n_blocks(), V.n_blocks());
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).copy_locally_owned_data_from(V.block(b));
        }

        virtual std::size_t
        memory_consumption() const
        {
          return MyMemoryConsumption::memory_consumption(blocks);
        }

        template <int dim, int spacedim>
        void
        save(const DoFHandler<dim, spacedim> &dof_handler,
             const std::string                file_name) const
        {
          // create unique map
          const auto unique_dof_map = create_unique_dof_map(dof_handler);

          // determine local size
          size_type local_size = 0;

          for (unsigned int b = 0; b < n_blocks(); ++b)
            local_size += block(b).locally_owned_size();

          // collect all values in a single vector
          std::vector<value_type> temp(local_size);

          for (unsigned int b = 0, c = 0; b < n_blocks(); ++b)
            for (unsigned int i = 0; i < block(b).locally_owned_size();
                 ++i, ++c)
              temp[c] = block(b).local_element(unique_dof_map[i]);

          // write to hard drive
          io(1, file_name, temp);
        }

        template <int dim, int spacedim>
        void
        load(const DoFHandler<dim, spacedim> &dof_handler,
             const std::string                file_name)
        {
          // create unique map
          const auto unique_dof_map = create_unique_dof_map(dof_handler);

          // determine local size
          size_type local_size = 0;

          for (unsigned int b = 0; b < n_blocks(); ++b)
            local_size += block(b).locally_owned_size();

          // read from hard drive
          std::vector<value_type> temp(local_size);
          io(0, file_name, temp);

          // split up single vector into blocks
          for (unsigned int b = 0, c = 0; b < n_blocks(); ++b)
            for (unsigned int i = 0; i < block(b).locally_owned_size();
                 ++i, ++c)
              block(b).local_element(unique_dof_map[i]) = temp[c];
        }

      private:
        unsigned int                            block_counter;
        std::vector<std::shared_ptr<BlockType>> blocks;

        types::global_dof_index size_;

        template <int dim, int spacedim = dim>
        void
        visit_cells_recursevely(
          const typename DoFHandler<dim, spacedim>::cell_iterator &cell,
          const IndexSet &           locally_owned_dofs,
          std::vector<bool> &        mask,
          std::vector<unsigned int> &result) const
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
              for (const auto child : cell->child_iterators())
                visit_cells_recursevely<dim, spacedim>(child,
                                                       locally_owned_dofs,
                                                       mask,
                                                       result);
            }
        }

        template <int dim, int spacedim>
        std::vector<unsigned int>
        create_unique_dof_map(
          const DoFHandler<dim, spacedim> &dof_handler) const
        {
          const auto &locally_owned_dofs = dof_handler.locally_owned_dofs();

          std::vector<unsigned int> result;
          result.reserve(locally_owned_dofs.n_elements());

          std::vector<bool> mask(locally_owned_dofs.n_elements(), false);

          for (const auto &cell : dof_handler.cell_iterators_on_level(0))
            visit_cells_recursevely<dim, spacedim>(cell,
                                                   locally_owned_dofs,
                                                   mask,
                                                   result);

          AssertThrow(result.size() == locally_owned_dofs.n_elements(),
                      ExcNotImplemented());

          return result;
        }

        void
        io(int type, std::string filename, std::vector<value_type> &src) const
        {
          const MPI_Comm comm = block(0).get_mpi_communicator();

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
          if (type == 0)
            ierr = MPI_File_open(
              comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
          else
            ierr = MPI_File_open(comm,
                                 filename.c_str(),
                                 MPI_MODE_CREATE | MPI_MODE_WRONLY,
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

          if (type == 0)
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
  }   // namespace LinearAlgebra

  template <typename Number>
  struct is_serial_vector<
    LinearAlgebra::distributed::DynamicBlockVector<Number>> : std::false_type
  {};

} // namespace dealii

#include <deal.II/lac/vector_memory.templates.h>
