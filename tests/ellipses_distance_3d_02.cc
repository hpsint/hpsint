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

// Arbitrarily located ellipsoids

#include <pf-applications/base/tensor.h>

#include <pf-applications/grain_tracker/ellipsoid.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 3;

  using Number = double;

  using namespace dealii;
  using namespace hpsint;
  using namespace GrainTracker;

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