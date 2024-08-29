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

#include <pf-applications/base/tensor.h>

#include <pf-applications/grain_tracker/ellipsoid.h>

#include <iostream>

constexpr unsigned int dim = 3;

using Number = double;

using namespace dealii;
using namespace hpsint;
using namespace GrainTracker;

void
test1()
{
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

  const auto res = distance(e1, e2);

  std::cout << "test1_dist   = " << res.first << std::endl;
  std::cout << "test1_n_iter = " << res.second << std::endl;
}

// This is the same geometry as in test 1 but using different parametrization
void
test2()
{
  std::array<Point<dim, Number>, dim> axes;
  for (unsigned int d = 0; d < dim; ++d)
    axes[d][d] = 1;

  // Ellipsoid 1
  const Point<dim, Number>      center1(2, 3, 1);
  const std::array<Number, dim> radii1{{2, 3, std::sqrt(18.)}};
  const Ellipsoid<dim, Number>  e1(center1, radii1, axes);

  // Ellipsoid 2
  const Point<dim, Number>      center2(-3, -2, -1);
  const std::array<Number, dim> radii2{{3, 2, std::sqrt(18.)}};
  const Ellipsoid<dim, Number>  e2(center2, radii2, axes);

  const auto res = distance(e1, e2);

  std::cout << "test2_dist   = " << res.first << std::endl;
  std::cout << "test2_n_iter = " << res.second << std::endl;
}

// Arbitrarily located ellipsoids
void
test3()
{
  // Ellipsoid 1
  const Point<dim, Number>      center1(-3, -2, -5);
  const std::array<Number, dim> radii1{{1, 3, 6}};

  const Point<dim, Number>     vec1(3, 7, -1);
  const Point<dim, Number>     axis1(unit_vector(vec1));
  const Number                 angle1 = 0.5;
  const Tensor<2, dim, Number> R1 =
    Physics::Transformations::Rotations::rotation_matrix_3d(axis1, angle1);
  const std::array<Point<dim, Number>, dim> axes1(tensor_to_point_array(R1));

  const Ellipsoid<dim, Number> e1(center1, radii1, axes1);

  // Ellipsoid 2
  const Point<dim, Number>      center2(5, 9, 11);
  const std::array<Number, dim> radii2{{5, 2, 3}};

  const Point<dim, Number>     vec2(6, 7, -2);
  const Point<dim, Number>     axis2(unit_vector(vec2));
  const Number                 angle2 = -0.2;
  const Tensor<2, dim, Number> R2 =
    Physics::Transformations::Rotations::rotation_matrix_3d(axis2, angle2);
  const std::array<Point<dim, Number>, dim> axes2(tensor_to_point_array(R2));

  const Ellipsoid<dim, Number> e2(center2, radii2, axes2);

  const auto res = distance(e1, e2, 1e-10, 300);

  std::cout << "test3_dist   = " << res.first << std::endl;
  std::cout << "test3_n_iter = " << res.second << std::endl;
}

int
main()
{
  test1();
  test2();
  test3();
}