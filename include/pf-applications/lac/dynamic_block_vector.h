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
        DynamicBlockVector(const unsigned int n = 0);

        void
        reinit(const unsigned int n);

        void
        reinit(const DynamicBlockVector<T> &V,
               const bool                   omit_zeroing_entries = false);

        /**
         * Blocks.
         */
        BlockType &
        block(const unsigned int i);

        const BlockType &
        block(const unsigned int i) const;

        unsigned int
        n_blocks() const;

        /**
         * Communication.
         */
        void
        update_ghost_values() const;

        void
        zero_out_ghost_values() const;

        bool
        has_ghost_elements() const;

        void
        compress(VectorOperation::values operation);

        /**
         * Computation.
         */
        T
        l2_norm() const;

        void
        add(const T a, const DynamicBlockVector<T> &V);

        void
        sadd(const T s, const DynamicBlockVector<T> &V);

        void
        sadd(const T s, const T a, const DynamicBlockVector<T> &V);

        void
        scale(const DynamicBlockVector<T> &V);

        void
        operator*=(const T factor);

        T
        add_and_dot(const T                      a,
                    const DynamicBlockVector<T> &V,
                    const DynamicBlockVector<T> &W);

        T
        operator*(const DynamicBlockVector<T> &V);
      };
    } // namespace distributed
  }   // namespace LinearAlgebra

  template <typename Number>
  struct is_serial_vector<
    LinearAlgebra::distributed::DynamicBlockVector<Number>> : std::false_type
  {};

} // namespace dealii