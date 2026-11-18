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

// Distance between shifted segments

#include <pf-applications/base/debug.h>

#include <pf-applications/grain_tracker/segment.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

template <int dim>
Segment<dim>
make_sphere_segment(const Point<dim> &center, const double radius)
{
  return Segment<dim>(center,
                      radius,
                      1.,
                      1.,
                      std::make_unique<RepresentationSpherical<dim>>(center,
                                                                     radius));
}

template <typename F>
bool
throws(F &&f)
{
  try
    {
      f();
    }
  catch (const ExceptionBase &)
    {
      return true;
    }

  return false;
}

int
main()
{
  std::cout << std::boolalpha << std::fixed << std::setprecision(4);

  std::cout << "Distance between shifted segments" << std::endl;

  const Tensor<1, 2> zero_shift{};
  const Tensor<1, 2> shift_left(
    make_array_view(std::vector<double>{-100., 0.}));
  const Tensor<1, 2> shift_right(
    make_array_view(std::vector<double>{100., 0.}));

  // Spheres
  {
    const auto segment_a = make_sphere_segment<2>(Point<2>(2., 50.), 3.);
    const auto segment_b = make_sphere_segment<2>(Point<2>(95., 50.), 2.);

    std::cout << "  spheres, no shift = " << segment_a.distance(segment_b)
              << std::endl;
    std::cout << "  spheres, zero shift = "
              << segment_a.distance(segment_b, zero_shift) << std::endl;
    std::cout << "  spheres, shift -100 = "
              << segment_a.distance(segment_b, shift_left) << std::endl;
    std::cout << "  spheres, reversed, shift 100 = "
              << segment_b.distance(segment_a, shift_right) << std::endl;
    std::cout << "  spheres, shift 100 = "
              << segment_a.distance(segment_b, shift_right) << std::endl;

    // The shift does not modify the segments
    debug::print_components(segment_b.get_center(),
                            "  spheres, center after",
                            std::cout);
  }

  // Spheres in 3D
  {
    const auto segment_a = make_sphere_segment<3>(Point<3>(0., 0., 1.), 1.);
    const auto segment_b = make_sphere_segment<3>(Point<3>(0., 0., 99.), 0.5);

    const Tensor<1, 3> shift(
      make_array_view(std::vector<double>{0., 0., -100.}));

    std::cout << "  spheres 3D, no shift = " << segment_a.distance(segment_b)
              << std::endl;
    std::cout << "  spheres 3D, shift -100 along z = "
              << segment_a.distance(segment_b, shift) << std::endl;
  }

  // Ellipses elongated along x
  {
    const std::array<double, 2>   radii{{3., 1.}};
    const std::array<Point<2>, 2> axes{{Point<2>(1., 0.), Point<2>(0., 1.)}};

    const auto make_ellipse_segment = [&](const Point<2> &center) {
      return Segment<2>(center,
                        3.,
                        M_PI * 3.,
                        1.,
                        std::make_unique<RepresentationElliptical<2>>(center,
                                                                      radii,
                                                                      axes));
    };

    const auto segment_a = make_ellipse_segment(Point<2>(2., 50.));
    const auto segment_b = make_ellipse_segment(Point<2>(95., 50.));

    std::cout << "  ellipses, no shift = " << segment_a.distance(segment_b)
              << std::endl;
    std::cout << "  ellipses, shift -100 = "
              << segment_a.distance(segment_b, shift_left) << std::endl;
  }

  // Wavefront: the shift has no effect
  {
    const std::map<std::pair<unsigned int, unsigned int>, double> distances{
      {{0, 2}, 4.5}};

    const Segment<2> segment_a(Point<2>(2., 50.),
                               3.,
                               1.,
                               1.,
                               std::make_unique<RepresentationWavefront<2>>(
                                 0, 1, Point<2>(2., 50.), distances));
    const Segment<2> segment_b(
      Point<2>(95., 50.),
      2.,
      1.,
      1.,
      std::make_unique<RepresentationWavefront<2>>(
        0,
        2,
        Point<2>(95., 50.),
        std::map<std::pair<unsigned int, unsigned int>, double>()));

    std::cout << "  wavefront, no shift = " << segment_a.distance(segment_b)
              << std::endl;
    std::cout << "  wavefront, shift -100 = "
              << segment_a.distance(segment_b, shift_left) << std::endl;
  }

  // Segments without representation can not be compared
  {
    const auto       segment = make_sphere_segment<2>(Point<2>(2., 50.), 3.);
    const Segment<2> empty;

    std::cout << "  uninitialized other, shifted: " << throws([&] {
      (void)segment.distance(empty, shift_left);
    }) << std::endl;
    std::cout << "  uninitialized other, zero shift: " << throws([&] {
      (void)segment.distance(empty, zero_shift);
    }) << std::endl;
    std::cout << "  uninitialized this, shifted: " << throws([&] {
      (void)empty.distance(segment, shift_left);
    }) << std::endl;
  }
}
