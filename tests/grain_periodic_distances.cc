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

// Distances between grains and the overlap check in a periodic domain

#include <pf-applications/grain_tracker/ellipsoid.h>
#include <pf-applications/grain_tracker/grain.h>
#include <pf-applications/grain_tracker/representation.h>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

constexpr int dim = 2;

Tensor<1, dim>
make_periods(const double px, const double py)
{
  Tensor<1, dim> periods;
  periods[0] = px;
  periods[1] = py;
  return periods;
}

Grain<dim>
make_grain(const unsigned int grain_id,
           const unsigned int order_parameter,
           const Point<dim>  &center,
           const double       radius)
{
  Grain<dim> grain(grain_id, order_parameter);
  grain.add_segment(center, radius, M_PI * radius * radius, 1.0);
  return grain;
}

void
report(const std::string    &label,
       const Grain<dim>     &grain_a,
       const Grain<dim>     &grain_b,
       const Tensor<1, dim> &periods,
       const double          buffer_ratio = 0.05)
{
  std::cout << label << std::endl;
  std::cout << "  periods = (" << periods[0] << ", " << periods[1] << ")"
            << std::endl;
  std::cout << "  lower bound    = "
            << grain_a.distance_lower_bound(grain_b, periods) << " / "
            << grain_b.distance_lower_bound(grain_a, periods) << std::endl;
  std::cout << "  distance       = " << grain_a.distance(grain_b, periods)
            << " / " << grain_b.distance(grain_a, periods) << std::endl;
  std::cout << "  overlaps       = "
            << grain_a.overlaps(grain_b, buffer_ratio, 0., periods) << " / "
            << grain_b.overlaps(grain_a, buffer_ratio, 0., periods)
            << std::endl;
}

void
test_grain_distance_2d()
{
  const auto no_periods = make_periods(0, 0);
  const auto periods_xy = make_periods(100, 100);
  const auto periods_x  = make_periods(100, 0);
  const auto periods_y  = make_periods(0, 100);

  /* Pair of grains of the same order parameter which collided in a simulation:
   * they are far from each other inside the domain but overlap through the
   * periodic boundary along x. */
  const auto grain_a = make_grain(13, 1, Point<dim>(4.341, 11.820), 5.870);
  const auto grain_b = make_grain(91, 1, Point<dim>(97.185, 9.056), 2.978);

  report("Collision across x: no periodicity", grain_a, grain_b, no_periods);
  report("Collision across x: periodic in x and y",
         grain_a,
         grain_b,
         periods_xy);
  report("Collision across x: periodic in x", grain_a, grain_b, periods_x);
  report("Collision across x: periodic in y only", grain_a, grain_b, periods_y);

  // The grains are far enough from each other in the periodic domain too
  const auto grain_c = make_grain(1, 0, Point<dim>(50., 50.), 5.);
  const auto grain_d = make_grain(2, 0, Point<dim>(10., 50.), 5.);

  report("Far grains: periodic in x and y", grain_c, grain_d, periods_xy);

  // The closest image is across the corner
  const auto grain_e = make_grain(3, 0, Point<dim>(2., 2.), 1.);
  const auto grain_f = make_grain(4, 0, Point<dim>(98., 98.), 1.);

  report("Corner: periodic in x and y", grain_e, grain_f, periods_xy);
  report("Corner: periodic in x only", grain_e, grain_f, periods_x);

  // Only the closest segment of a grain having several of them matters
  Grain<dim> grain_g(5, 0);
  grain_g.add_segment(Point<dim>(50., 50.), 5., M_PI * 25., 1.0);
  grain_g.add_segment(Point<dim>(2., 50.), 2., M_PI * 4., 1.0);

  const auto grain_h = make_grain(6, 0, Point<dim>(95., 50.), 2.);

  report("Multiple segments: periodic in x", grain_g, grain_h, periods_x);

  // The nearest neighbor distance defines the transfer buffer
  for (const auto &periods : {no_periods, periods_xy})
    {
      auto grain_a_copy = grain_a;
      grain_a_copy.add_neighbor(grain_b, periods);

      std::cout << "Transfer buffer, periods = (" << periods[0] << ", "
                << periods[1] << ") : " << grain_a_copy.transfer_buffer()
                << std::endl;
    }

  /* Ellipses having their long axes along x. The distance between them is
   * 1 across the periodic boundary and 87 inside the domain. */
  std::array<Point<dim>, dim> axes;
  for (unsigned int d = 0; d < dim; ++d)
    axes[d][d] = 1;

  const auto make_elliptical_grain = [&](const unsigned int grain_id,
                                         const Point<dim>  &center) {
    const std::array<double, dim> radii{{3., 1.}};

    Grain<dim> grain(grain_id, 0);
    grain.add_segment(Segment<dim>(
      center,
      3.,
      M_PI * 3.,
      1.0,
      std::make_unique<RepresentationElliptical<dim>>(center, radii, axes)));
    return grain;
  };

  const auto ellipse_a = make_elliptical_grain(7, Point<dim>(2., 50.));
  const auto ellipse_b = make_elliptical_grain(8, Point<dim>(95., 50.));

  report("Ellipses: no periodicity", ellipse_a, ellipse_b, no_periods);
  report("Ellipses: periodic in x", ellipse_a, ellipse_b, periods_x);
}

void
test_grain_distance_3d()
{
  std::cout << "Grain distance in 3D" << std::endl;

  Grain<3> grain_a(0, 0);
  grain_a.add_segment(Point<3>(1., 1., 1.), 1., 1., 1.);

  Grain<3> grain_b(1, 0);
  grain_b.add_segment(Point<3>(9., 19., 5.), 1., 1., 1.);

  // Periodic along x and y only
  const Tensor<1, 3> periods(
    make_array_view(std::vector<double>{10., 20., 0.}));

  std::cout << "  no periods = " << grain_a.distance(grain_b) << std::endl;
  std::cout << "  periodic in x and y = " << grain_a.distance(grain_b, periods)
            << std::endl;
  std::cout << "  lower bound, periodic in x and y = "
            << grain_a.distance_lower_bound(grain_b, periods) << std::endl;

  // The closest image is across the corner of the domain
  Grain<3> grain_c(2, 0);
  grain_c.add_segment(Point<3>(99., 99., 99.), 1., 1., 1.);

  const Tensor<1, 3> periods_xyz(
    make_array_view(std::vector<double>{100., 100., 100.}));

  std::cout << "  corner, periodic in x, y and z = "
            << grain_a.distance(grain_c, periods_xyz) << std::endl;

  /* Ellipsoids having one long semi-axis of 3 and two short ones of 1. Their
   * centers are 93 apart inside the domain and 7 across the periodic boundary
   * along x. */
  const Tensor<1, 3> periods_x(
    make_array_view(std::vector<double>{100., 0., 0.}));

  const auto make_ellipsoid_grain = [](const unsigned int             grain_id,
                                       const Point<3>                &center,
                                       const std::array<Point<3>, 3> &axes) {
    const std::array<double, 3> radii{{3., 1., 1.}};

    Grain<3> grain(grain_id, 0);
    grain.add_segment(Segment<3>(
      center,
      3.,
      1.,
      1.0,
      std::make_unique<RepresentationElliptical<3>>(center, radii, axes)));
    return grain;
  };

  const auto report_ellipsoids = [&](const std::string             &label,
                                     const std::array<Point<3>, 3> &axes) {
    const auto ellipsoid_a =
      make_ellipsoid_grain(3, Point<3>(2., 50., 50.), axes);
    const auto ellipsoid_b =
      make_ellipsoid_grain(4, Point<3>(95., 50., 50.), axes);

    std::cout << "  " << label
              << ", no periods = " << ellipsoid_a.distance(ellipsoid_b)
              << std::endl;
    std::cout << "  " << label << ", periodic in x = "
              << ellipsoid_a.distance(ellipsoid_b, periods_x) << std::endl;
  };

  // The long axes are along x, then along y to check that the axes are kept
  report_ellipsoids(
    "ellipsoids along x",
    {{Point<3>(1., 0., 0.), Point<3>(0., 1., 0.), Point<3>(0., 0., 1.)}});
  report_ellipsoids(
    "ellipsoids along y",
    {{Point<3>(0., 1., 0.), Point<3>(1., 0., 0.), Point<3>(0., 0., 1.)}});
}

int
main()
{
  std::cout << std::boolalpha << std::fixed << std::setprecision(4);

  test_grain_distance_2d();
  test_grain_distance_3d();
}
