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

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>

using namespace dealii;

template <typename VectorType, typename Enable = void>
struct is_dealii_compatible_vector;

template <typename VectorType>
struct is_dealii_compatible_vector<
  VectorType,
  typename std::enable_if<!internal::is_block_vector<VectorType>>::type>
{
  static constexpr bool value = std::is_same<
    VectorType,
    LinearAlgebra::distributed::Vector<typename VectorType::value_type,
                                       MemorySpace::Host>>::value;
};

template <typename VectorType>
struct is_dealii_compatible_vector<
  VectorType,
  typename std::enable_if<internal::is_block_vector<VectorType>>::type>
{
  static constexpr bool value = std::is_same<
    typename VectorType::BlockType,
    LinearAlgebra::distributed::Vector<typename VectorType::value_type,
                                       MemorySpace::Host>>::value;
};

template <typename VectorType,
          std::enable_if_t<!is_dealii_compatible_vector<VectorType>::value,
                           VectorType> * = nullptr>
void
test(const VectorType &)
{
  std::cout << "Fail!" << std::endl;
}

template <typename VectorType,
          std::enable_if_t<is_dealii_compatible_vector<VectorType>::value,
                           VectorType> * = nullptr>
void
test(const VectorType &)
{
  std::cout << "Success!" << std::endl;
}

int
main()
{
  {
    LinearAlgebra::distributed::Vector<double> vec;
    test(vec);
  }

  {
    LinearAlgebra::distributed::BlockVector<double> vec;
    test(vec);
  }

  {
    LinearAlgebra::distributed::DynamicBlockVector<double> vec;
    test(vec);
  }

  {
    Vector<double> vec;
    test(vec);
  }
}
