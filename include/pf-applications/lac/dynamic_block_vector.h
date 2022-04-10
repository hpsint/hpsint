#pragma once

#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/vector_operation.h>

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
        {
          reinit(n);
        }

        explicit DynamicBlockVector(const DynamicBlockVector<T> &V)
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
              block(b) = V.block(b);
            }

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

        void
        add(const T a, const DynamicBlockVector<T> &V)
        {
          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b).add(a, V.block(b));
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

      private:
        unsigned int                            block_counter;
        std::vector<std::shared_ptr<BlockType>> blocks;
      };
    } // namespace distributed
  }   // namespace LinearAlgebra

  template <typename Number>
  struct is_serial_vector<
    LinearAlgebra::distributed::DynamicBlockVector<Number>> : std::false_type
  {};

} // namespace dealii

#include <deal.II/lac/vector_memory.templates.h>
