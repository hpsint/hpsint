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

#include <pf-applications/grain_tracker/ellipsoid.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 3;

  using Number = double;

  using namespace dealii;
  using namespace GrainTracker;

  // Ellipse center = (2, 3, 1);  radii = (2, 3, sqrt(18))
  const Number                          data_a1[] = {18, 8, 4, 0, 0, 0};
  const SymmetricTensor<2, dim, Number> A1(data_a1);
  const Number                          data_b1[dim] = {-36, -24, -4};
  const Tensor<1, dim, Number>          b1(data_b1);
  const Number                          alpha1 = 38;
  const Ellipsoid<dim, Number>          e1(A1, b1, alpha1);

  // Ellipse center = (-3, -2, -1);  radii = (3, 2, sqrt(18))
  const Number                          data_a2[] = {8, 18, 4, 0, 0, 0};
  const SymmetricTensor<2, dim, Number> A2(data_a2);
  const Number                          data_b2[dim] = {24, 36, 4};
  const Tensor<1, dim, Number>          b2(data_b2);
  const Number                          alpha2 = 38;
  const Ellipsoid<dim, Number>          e2(A2, b2, alpha2);

  const auto [dist_tf, iter_tf, status_tf] =
    distance(e1, e2, true, false);
  const auto [dist_ft, iter_ft, status_ft] =
    distance(e1, e2, false, true);

  std::cout << std::setprecision(15);
  std::cout << "dist_tf   = " << dist_tf << std::endl;
  std::cout << "n_iter_tf = " << iter_tf << std::endl;
  std::cout << "dist_ft   = " << dist_ft << std::endl;
  std::cout << "n_iter_ft = " << iter_ft << std::endl;
}