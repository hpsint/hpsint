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
  constexpr unsigned int dim = 2;

  using Number = double;

  using namespace dealii;
  using namespace GrainTracker;

  // Ellipse center = (2, 3);  radii = (2, 3)
  const Number                          data_a1[] = {18, 8, 0};
  const SymmetricTensor<2, dim, Number> A1(data_a1);
  const Number                          data_b1[dim] = {-36, -24};
  const Tensor<1, dim, Number>          b1(data_b1);
  const Number                          alpha1 = 36;
  const Ellipsoid<dim, Number>          e1(A1, b1, alpha1);

  // Ellipse center = (-3, -2);  radii = (3, 2)
  const Number                          data_a2[] = {8, 18, 0};
  const SymmetricTensor<2, dim, Number> A2(data_a2);
  const Number                          data_b2[dim] = {24, 36};
  const Tensor<1, dim, Number>          b2(data_b2);
  const Number                          alpha2 = 36;
  const Ellipsoid<dim, Number>          e2(A2, b2, alpha2);

  // Ellipse center = (2, 3); tilted, equation: 2x^2 + y^2 = 2(x + y + xy -1)
  const Number                          data_a3[] = {4, 2, -2};
  const SymmetricTensor<2, dim, Number> A3(data_a3);
  const Number                          data_b3[dim] = {-2, -2};
  const Tensor<1, dim, Number>          b3(data_b3);
  const Number                          alpha3 = 1;
  const Ellipsoid<dim, Number>          e3(A3, b3, alpha3);

  const auto res12 = distance(e1, e2);

  std::cout << "dist_e1_e2   = " << res12.first << std::endl;
  std::cout << "n_iter_e1_e2 = " << res12.second << std::endl;

  const auto res32 = distance(e3, e2);

  std::cout << "dist_e3_e2   = " << res32.first << std::endl;
  std::cout << "n_iter_e3_e2 = " << res32.second << std::endl;
}