// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>

#include <pf-applications/numerics/power_helper.h>

#include <array>
#include <vector>

using namespace dealii;
using namespace hpsint;

int
main()
{
  const auto dump = [](const auto &begin, const auto &end) {
    std::copy(begin, end, std::ostream_iterator<double>(std::cout, " "));
    std::cout << std::endl;
  };

  // Non-optimized versions
  constexpr unsigned int n_grains = 4;

  std::cout << "Generalized versions:" << std::endl;

  // Test an array
  std::array<double, n_grains> arr = {{3., 4., 2., -5.}};
  std::cout << "std::array = ";
  dump(arr.begin(), arr.end());
  std::cout << "p=1: " << PowerHelper<n_grains, 1>::power_sum(arr) << std::endl;
  std::cout << "p=2: " << PowerHelper<n_grains, 2>::power_sum(arr) << std::endl;
  std::cout << "p=3: " << PowerHelper<n_grains, 3>::power_sum(arr) << std::endl;
  std::cout << "p=4: " << PowerHelper<n_grains, 4>::power_sum(arr) << std::endl;

  // Test a first-order tenor
  const Tensor<1, n_grains, double> ten(
    ArrayView<double>(arr.data(), n_grains));
  std::cout << "Tensor = ";
  dump(hpsint::cbegin(ten), hpsint::cend(ten));
  std::cout << "p=1: " << PowerHelper<n_grains, 1>::power_sum(ten) << std::endl;
  std::cout << "p=2: " << PowerHelper<n_grains, 2>::power_sum(ten) << std::endl;
  std::cout << "p=3: " << PowerHelper<n_grains, 3>::power_sum(ten) << std::endl;
  std::cout << "p=4: " << PowerHelper<n_grains, 4>::power_sum(ten) << std::endl;

  // Test raw pointers
  const auto ptr = arr.data();
  std::cout << "pointer data = ";
  dump(ptr, ptr + n_grains);
  std::cout << "p=1: " << PowerHelper<n_grains, 1>::power_sum(ptr) << std::endl;
  std::cout << "p=2: " << PowerHelper<n_grains, 2>::power_sum(ptr) << std::endl;
  std::cout << "p=3: " << PowerHelper<n_grains, 3>::power_sum(ptr) << std::endl;
  std::cout << "p=4: " << PowerHelper<n_grains, 4>::power_sum(ptr) << std::endl;

  // Test vector
  const std::vector<double> vec(arr.begin(), arr.end());
  std::cout << "std::vector = ";
  dump(vec.begin(), vec.end());
  std::cout << "p=1: " << PowerHelper<n_grains, 1>::power_sum(vec) << std::endl;
  std::cout << "p=2: " << PowerHelper<n_grains, 2>::power_sum(vec) << std::endl;
  std::cout << "p=3: " << PowerHelper<n_grains, 3>::power_sum(vec) << std::endl;
  std::cout << "p=4: " << PowerHelper<n_grains, 4>::power_sum(vec) << std::endl;

  // Optimized versions
  std::cout << "Optimized versions:" << std::endl;

  std::array<double, 2> arr_small = {{3., 4.}};
  std::cout << "std::array = ";
  dump(arr_small.begin(), arr_small.end());
  std::cout << "p=2: " << PowerHelper<2, 2>::power_sum(arr_small) << std::endl;
  std::cout << "p=3: " << PowerHelper<2, 3>::power_sum(arr_small) << std::endl;

  const Tensor<1, 2, double> ten_small(ArrayView<double>(arr_small.data(), 2));
  std::cout << "Tensor = ";
  dump(hpsint::cbegin(ten_small), hpsint::cend(ten_small));
  std::cout << "p=2: " << PowerHelper<2, 2>::power_sum(ten_small) << std::endl;
  std::cout << "p=3: " << PowerHelper<2, 3>::power_sum(ten_small) << std::endl;

  const auto ptr_small = arr_small.data();
  std::cout << "pointer data = ";
  dump(ptr, ptr + 2);
  std::cout << "p=2: " << PowerHelper<2, 2>::power_sum(ptr_small) << std::endl;
  std::cout << "p=3: " << PowerHelper<2, 3>::power_sum(ptr_small) << std::endl;
}