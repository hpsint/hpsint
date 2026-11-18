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

// Translated copies of the representations: clone_shifted()

#include <pf-applications/base/debug.h>

#include <pf-applications/grain_tracker/representation.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

int
main()
{
  std::cout << std::boolalpha << std::fixed << std::setprecision(4);

  std::cout << "Clone shifted representations" << std::endl;

  const std::vector<double> shift{10., -5.};

  // Spherical
  {
    const RepresentationSpherical<2> sphere(Point<2>(1., 2.), 3.);

    const auto  shifted = sphere.clone_shifted(shift);
    const auto *result =
      dynamic_cast<const RepresentationSpherical<2> *>(shifted.get());
    AssertThrow(result, ExcMessage("Wrong type of the shifted clone"));

    debug::print_components(result->center,
                            "  spherical: shifted center",
                            std::cout);
    std::cout << "  spherical: shifted radius = " << result->radius
              << std::endl;
    debug::print_components(sphere.center,
                            "  spherical: original center",
                            std::cout);
  }

  // Elliptical, the axes are rotated to check that they are preserved
  {
    const std::array<double, 2>   radii{{3., 1.}};
    const std::array<Point<2>, 2> axes{{Point<2>(0., 1.), Point<2>(-1., 0.)}};
    const RepresentationElliptical<2> ellipse(Point<2>(1., 2.), radii, axes);

    const auto  shifted = ellipse.clone_shifted(shift);
    const auto *result =
      dynamic_cast<const RepresentationElliptical<2> *>(shifted.get());
    AssertThrow(result, ExcMessage("Wrong type of the shifted clone"));

    debug::print_components(result->ellipsoid.get_center(),
                            "  elliptical: shifted center",
                            std::cout);
    debug::print_components(result->ellipsoid.get_radii(),
                            "  elliptical: shifted radii",
                            std::cout);
    debug::print_components(result->ellipsoid.get_axes()[0],
                            "  elliptical: shifted axis 0",
                            std::cout);
    debug::print_components(result->ellipsoid.get_axes()[1],
                            "  elliptical: shifted axis 1",
                            std::cout);
    debug::print_components(ellipse.ellipsoid.get_center(),
                            "  elliptical: original center",
                            std::cout);
  }

  // Wavefront: distances are precomputed and do not depend on the location
  {
    const std::map<std::pair<unsigned int, unsigned int>, double> distances{
      {{0, 2}, 4.5}, {{1, 3}, 7.}};

    const RepresentationWavefront<2> wavefront(0,
                                               1,
                                               Point<2>(1., 2.),
                                               distances);
    const RepresentationWavefront<2> other(0, 2, Point<2>(5., 5.), {});

    const auto  shifted = wavefront.clone_shifted(shift);
    const auto *result =
      dynamic_cast<const RepresentationWavefront<2> *>(shifted.get());
    AssertThrow(result, ExcMessage("Wrong type of the shifted clone"));

    debug::print_components(result->center, "  wavefront: center", std::cout);
    std::cout << "  wavefront: op and index = (" << result->op_and_index.first
              << ", " << result->op_and_index.second << ")" << std::endl;
    std::cout << "  wavefront: number of distances = "
              << result->distances.size() << std::endl;
    std::cout << "  wavefront: distance = "
              << static_cast<const Representation &>(wavefront).distance(other)
              << std::endl;
    std::cout << "  wavefront: shifted distance = " << shifted->distance(other)
              << std::endl;
  }
}
