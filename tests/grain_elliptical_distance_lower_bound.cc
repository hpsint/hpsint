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
#include <pf-applications/grain_tracker/grain.h>
#include <pf-applications/grain_tracker/representation.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 2;

  using Number = double;

  using namespace dealii;
  using namespace GrainTracker;

  std::array<Point<dim, Number>, dim> global_axes;
  for (unsigned int d = 0; d < dim; ++d)
    global_axes[d][d] = 1;

  // Ellipse center = (2, 3);  radii = (2, 3)
  constexpr Point<dim, Number>      center1(2, 3);
  constexpr std::array<Number, dim> radii1{{2, 3}};

  RepresentationElliptical<dim> r1_el(center1, radii1, global_axes);
  RepresentationSpherical<dim>  r1_sp(center1,
                                     *std::max_element(radii1.cbegin(),
                                                       radii1.cend()));

  // Ellipse center = (-3, -2);  radii = (3, 2)
  constexpr Point<dim, Number>      center2(-3, -2);
  constexpr std::array<Number, dim> radii2{{3, 2}};

  RepresentationElliptical<dim> r2_el(center2, radii2, global_axes);
  RepresentationSpherical<dim>  r2_sp(center2,
                                     *std::max_element(radii2.cbegin(),
                                                       radii2.cend()));

  Grain<dim> grain1(0, 0);
  grain1.add_segment(
    Segment(center1,
            *std::max_element(radii1.cbegin(), radii1.cend()),
            0.0,
            1.0,
            std::make_unique<RepresentationElliptical<dim>>(r1_el)));

  Grain<dim> grain2(1, 0);
  grain2.add_segment(
    Segment(center2,
            *std::max_element(radii2.cbegin(), radii2.cend()),
            0.0,
            1.0,
            std::make_unique<RepresentationElliptical<dim>>(r2_el)));

  const auto dist_el1_el2             = r1_el.distance(r2_el);
  const auto dist_sp1_sp2             = r1_sp.distance(r2_sp);
  const auto dist_gr1_gr2             = grain1.distance(grain2);
  const auto dist_lower_bound_gr1_gr2 = grain1.distance_lower_bound(grain2);

  std::cout << std::setprecision(15);
  std::cout << "distance rep_el1_rep_el2           = " << dist_el1_el2
            << std::endl;
  std::cout << "distance rep_sp1_rep_sp2           = " << dist_sp1_sp2
            << std::endl;
  std::cout << "distance gr_el1_gr_el2             = " << dist_gr1_gr2
            << std::endl;
  std::cout << "distance_lower_bound gr_el1_gr_el2 = "
            << dist_lower_bound_gr1_gr2 << std::endl;
}