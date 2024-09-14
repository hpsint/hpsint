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

#include <deal.II/physics/transformations.h>

#include <pf-applications/base/tensor.h>

#include <pf-applications/grain_tracker/representation.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 3;

  using Number = double;

  using namespace dealii;
  using namespace hpsint;
  using namespace GrainTracker;

  Number a = 5;
  Number b = 3;
  Number c = 2;

  Number volume = 4. / 3. * M_PI * a * b * c;

  Number Ixx = volume / 5. * (b * b + c * c);
  Number Iyy = volume / 5. * (a * a + c * c);
  Number Izz = volume / 5. * (a * a + b * b);

  const Number data_I_principal[] = {Ixx, Iyy, Izz, 0, 0, 0};
  const SymmetricTensor<2, dim, Number> I_principal(data_I_principal);

  const Point<3, Number>       vec(3, 7, -1);
  const Point<3, Number>       axis(unit_vector(vec));
  const Number                 angle = 0.5;
  const Tensor<2, dim, Number> R =
    Physics::Transformations::Rotations::rotation_matrix_3d(axis, angle);

  const Point<dim, Number>      center{3, 5, 7};
  const std::array<Number, dim> principal_moments{{Ixx, Iyy, Izz}};

  const SymmetricTensor<2, dim, Number> I_current =
    Physics::Transformations::basis_transformation(I_principal, R);

  RepresentationElliptical<dim> representation(center, volume, I_current);

  std::cout << "Input data:" << std::endl;
  std::cout << "center  = " << center << std::endl;
  std::cout << "radii   = " << a << " " << b << " " << c << std::endl;
  std::cout << "inertia = ";
  std::copy(principal_moments.begin(),
            principal_moments.end(),
            std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "axes    = " << R << std::endl;
  std::cout << "volume  = " << volume << std::endl;
  std::cout << std::endl;

  std::cout << "Representation data:" << std::endl;
  representation.print(std::cout);
  std::cout << std::endl;
}