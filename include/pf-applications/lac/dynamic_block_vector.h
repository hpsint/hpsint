#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>

namespace dealii
{
  namespace LinearAlgebra
  {
    namespace distributed
    {
      template <typename T>
      using DynamicBlockVector = BlockVector<T>;
    }
  } // namespace LinearAlgebra
} // namespace dealii