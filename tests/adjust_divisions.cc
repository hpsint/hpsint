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

#include <pf-applications/base/debug.h>

#include <pf-applications/grid/grid_tools.h>

using namespace hpsint;

void
run_test(std::vector<unsigned int> subdivisions, const unsigned int max_prime)
{
  std::cout << "max_prime = " << max_prime << std::endl;
  std::cout << "subdivisions initial: " << debug::to_string(subdivisions)
            << std::endl;

  const unsigned int n_refinements_base =
    hpsint::internal::adjust_divisions_to_primes(max_prime, subdivisions);

  std::cout << "subdivisions reduced: " << debug::to_string(subdivisions)
            << std::endl;

  std::transform(subdivisions.cbegin(),
                 subdivisions.cend(),
                 subdivisions.begin(),
                 [n_refinements_base](const auto d) {
                   return d * std::pow(2, n_refinements_base);
                 });

  std::cout << "subdivisions refined: " << debug::to_string(subdivisions)
            << std::endl;

  std::cout << "n_refinements_base = " << n_refinements_base << std::endl;
  std::cout << std::endl;
}

int
main()
{
  run_test({28, 27, 29}, 30);
  run_test({28, 27, 29}, 20);
  run_test({33, 34, 35}, 20);
  run_test({42, 41, 43}, 20);
}