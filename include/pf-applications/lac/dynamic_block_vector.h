#pragma once

#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
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

        template <typename Iterator>
        explicit DynamicBlockVector(Iterator begin, Iterator end)
        {
          blocks.assign(begin, end);

          size_ = 0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            size_ += block(b).size();
        }

        DynamicBlockVector<T> &
        operator=(const DynamicBlockVector<T> &V)
        {
          blocks.resize(V.n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            {
              if (blocks[b] == nullptr)
                blocks[b] = std::make_shared<BlockType>();
              block(b).reinit(V.block(b), true);
              block(b) = V.block(b);

              if (V.block(b).has_ghost_elements())
                block(b).update_ghost_values();
            }

          size_ = 0;
          for (unsigned int b = 0; b < n_blocks(); ++b)
            size_ += block(b).size();

          return *this;
        }

        void
        reinit(const unsigned int n, const bool omit_zeroing_entries = false)
        {
          const unsigned int old_blocks_size = blocks.size();

          blocks.resize(n);

          for (unsigned int b = old_blocks_size; b < n_blocks(); ++b)
            {
              if (blocks[b] == nullptr)
                blocks[b] = std::make_shared<BlockType>();

              if (old_blocks_size != 0)
                {
                  block(b).reinit(block(0), omit_zeroing_entries);
                  if (block(0).has_ghost_elements())
                    block(b).update_ghost_values();
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
          blocks.resize(V.n_blocks());

          for (unsigned int b = 0; b < n_blocks(); ++b)
            {
              if (blocks[b] == nullptr)
                blocks[b] = std::make_shared<BlockType>();
              block(b).reinit(V.block(b), omit_zeroing_entries);
              if (block(0).has_ghost_elements())
                block(b).update_ghost_values();
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

        std::shared_ptr<BlockType>
        block_ptr(const unsigned int i)
        {
          AssertIndexRange(i, n_blocks());
          return blocks[i];
        }

        std::shared_ptr<BlockType>
        block_ptr(const unsigned int i) const
        {
          AssertIndexRange(i, n_blocks());
          return blocks[i];
        }

        unsigned int
        n_blocks() const
        {
          return blocks.size();
        }

        /* Move block to a new place */
        void
        move_block(const unsigned int from, const unsigned int to)
        {
          AssertIndexRange(from, n_blocks());
          AssertIndexRange(to, n_blocks());

          auto tmp = blocks[from];

          if (from > to)
            for (unsigned int i = from; i > to; --i)
              blocks[i] = blocks[i - 1];
          else if (from < to)
            for (unsigned int i = from; i < to; ++i)
              blocks[i] = blocks[i + 1];

          blocks[to] = tmp;
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
        equ(const T a, const DynamicBlockVector<T> &v)
        {
          *this = v;

          for (unsigned int b = 0; b < n_blocks(); ++b)
            block(b) *= a;
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
        operator*(const DynamicBlockVector<T> &V) const
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

        void
        swap(DynamicBlockVector &V)
        {
          std::swap(this->blocks, V.blocks);
          std::swap(this->size_, V.size_);
        }

        static constexpr unsigned int communication_block_size = 0;

      private:
        std::vector<std::shared_ptr<BlockType>> blocks;

        types::global_dof_index size_;
      };
    } // namespace distributed
  }   // namespace LinearAlgebra

  template <typename Number>
  struct is_serial_vector<
    LinearAlgebra::distributed::DynamicBlockVector<Number>> : std::false_type
  {};

} // namespace dealii

#include <deal.II/lac/vector_memory.templates.h>

namespace Sintering
{
  using namespace dealii;


  namespace internal
  {
    template <typename Number>
    unsigned int
    n_blocks(const LinearAlgebra::distributed::Vector<Number> &)
    {
      return 1;
    }

    template <typename Number>
    unsigned int
    n_blocks(const LinearAlgebra::distributed::BlockVector<Number> &vector)
    {
      return vector.n_blocks();
    }

    template <typename Number>
    unsigned int
    n_blocks(
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &vector)
    {
      return vector.n_blocks();
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::Vector<Number> &vector,
          const unsigned int                          b)
    {
      AssertThrow(b == 0, ExcInternalError());
      return vector;
    }

    template <typename Number>
    const LinearAlgebra::distributed::Vector<Number> &
    block(const LinearAlgebra::distributed::Vector<Number> &vector,
          const unsigned int                                b)
    {
      AssertThrow(b == 0, ExcInternalError());
      return vector;
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::BlockVector<Number> &vector,
          const unsigned int                               b)
    {
      return vector.block(b);
    }

    template <typename Number>
    const LinearAlgebra::distributed::Vector<Number> &
    block(const LinearAlgebra::distributed::BlockVector<Number> &vector,
          const unsigned int                                     b)
    {
      return vector.block(b);
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::DynamicBlockVector<Number> &vector,
          const unsigned int                                      b)
    {
      return vector.block(b);
    }

    template <typename Number>
    const LinearAlgebra::distributed::Vector<Number> &
    block(const LinearAlgebra::distributed::DynamicBlockVector<Number> &vector,
          const unsigned int                                            b)
    {
      return vector.block(b);
    }

  } // namespace internal
} // namespace Sintering
