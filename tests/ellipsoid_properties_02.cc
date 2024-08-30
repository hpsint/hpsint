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

  std::array<Point<dim, Number>, dim> axes;
  for (unsigned int d = 0; d < dim; ++d)
    axes[d][d] = 1;

  const Point<dim, Number>      center(2, 3, 1);
  const std::array<Number, dim> radii{{6, 7, 8}};
  const Ellipsoid<dim, Number>  ellipsoid(center, radii, axes);

  std::cout << "Input data:" << std::endl;
  std::cout << "center  = " << center << std::endl;
  std::cout << "radii = ";
  std::copy(radii.begin(),
            radii.end(),
            std::ostream_iterator<Number>(std::cout, " "));
  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Ellipsoid data:" << std::endl;
  std::cout << "A       = " << ellipsoid.A << std::endl;
  std::cout << "b       = " << ellipsoid.b << std::endl;
  std::cout << "alpha   = " << ellipsoid.alpha << std::endl;
  std::cout << "radii   = ";
  std::copy(ellipsoid.radii.begin(),
            ellipsoid.radii.end(),
            std::ostream_iterator<Number>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "norm    = " << ellipsoid.norm << std::endl;
  std::cout << "gamma   = " << ellipsoid.gamma << std::endl;
  std::cout << "center  = " << ellipsoid.center << std::endl;
}