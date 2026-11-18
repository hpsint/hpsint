// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

// Translation to the closest periodic image: minimum_image_shift()

#include <deal.II/base/array_view.h>

#include <pf-applications/base/debug.h>
#include <pf-applications/base/geometry.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
using namespace hpsint;

template <int dim>
void
test_minimum_image_shift(const std::string         &label,
                         const Point<dim>          &a,
                         const Point<dim>          &b,
                         const std::vector<double> &periods)
{
  const Tensor<1, dim> periods_tensor(make_array_view(periods));

  debug::print_components(minimum_image_shift(a, b, periods_tensor),
                          "  " + label,
                          std::cout);
}

int
main()
{
  std::cout << std::boolalpha << std::fixed << std::setprecision(4);

  std::cout << "Minimum image shift" << std::endl;
  test_minimum_image_shift<2>("right image",
                              Point<2>(1., 5.),
                              Point<2>(9., 7.),
                              {10., 0.});
  test_minimum_image_shift<2>("left image",
                              Point<2>(9., 5.),
                              Point<2>(1., 5.),
                              {10., 0.});
  test_minimum_image_shift<2>("same cell",
                              Point<2>(4., 5.),
                              Point<2>(6., 5.),
                              {10., 0.});
  test_minimum_image_shift<2>("several periods",
                              Point<2>(1., 5.),
                              Point<2>(31., 5.),
                              {10., 0.});
  test_minimum_image_shift<2>("periodic in x and y",
                              Point<2>(1., 1.),
                              Point<2>(9., 19.),
                              {10., 20.});
  test_minimum_image_shift<2>("non-periodic",
                              Point<2>(1., 1.),
                              Point<2>(99., 99.),
                              {0., 0.});
  test_minimum_image_shift<3>("3D",
                              Point<3>(1., 1., 1.),
                              Point<3>(9., 19., 50.),
                              {10., 20., 0.});
}
