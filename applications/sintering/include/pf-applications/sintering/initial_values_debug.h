// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the hpsint authors
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

#pragma once

#include <pf-applications/sintering/initial_values.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim>
  class InitialValuesDebug : public InitialValues<dim>
  {
  public:
    InitialValuesDebug()
      : InitialValues<dim>(1.0)
    {}

    double
    do_value(const Point<dim> &p, const unsigned int component) const final
    {
      std::vector<double> vals;

      const double tol = 1e-9;

      if constexpr (dim == 2)
        {
          Point<dim> p1(0, 0);
          Point<dim> p2(1, 0);
          Point<dim> p3(2, 0);
          Point<dim> p4(0, 1);
          Point<dim> p5(1, 1);
          Point<dim> p6(2, 1);

          if (p.distance(p1) < tol)
            vals = {1.0, 0.1, 1.0, 0.0};
          else if (p.distance(p2) < tol)
            vals = {0.8, 0.2, 0.4, 0.4};
          else if (p.distance(p3) < tol)
            vals = {1.0, 0.3, 0.0, 1.0};
          else if (p.distance(p4) < tol)
            vals = {1.0, 0.4, 1.0, 0.0};
          else if (p.distance(p5) < tol)
            vals = {0.8, 0.5, 0.4, 0.4};
          else if (p.distance(p6) < tol)
            vals = {1.0, 0.6, 0.0, 1.0};
          else
            {
              std::cout << "Point = " << p << std::endl;
              throw std::runtime_error("Wrong point!");
            }
        }
      else
        {
          Point<dim> p1(0, 0, 0);
          Point<dim> p2(1, 0, 0);
          Point<dim> p3(2, 0, 0);
          Point<dim> p4(0, 1, 0);
          Point<dim> p5(1, 1, 0);
          Point<dim> p6(2, 1, 0);
          Point<dim> p7(0, 0, 1);
          Point<dim> p8(1, 0, 1);
          Point<dim> p9(2, 0, 1);
          Point<dim> p10(0, 1, 1);
          Point<dim> p11(1, 1, 1);
          Point<dim> p12(2, 1, 1);

          if (p.distance(p1) < tol || p.distance(p7) < tol)
            vals = {1.0, 0.1, 1.0, 0.0};
          else if (p.distance(p2) < tol || p.distance(p8) < tol)
            vals = {0.8, 0.2, 0.4, 0.4};
          else if (p.distance(p3) < tol || p.distance(p9) < tol)
            vals = {1.0, 0.3, 0.0, 1.0};
          else if (p.distance(p4) < tol || p.distance(p10) < tol)
            vals = {1.0, 0.4, 1.0, 0.0};
          else if (p.distance(p5) < tol || p.distance(p11) < tol)
            vals = {0.8, 0.5, 0.4, 0.4};
          else if (p.distance(p6) < tol || p.distance(p12) < tol)
            vals = {1.0, 0.6, 0.0, 1.0};
          else
            {
              std::cout << "Point = " << p << std::endl;
              throw std::runtime_error("Wrong point!");
            }
        }

      return component < 4 ? vals[component] : 0.0;
    }

    std::pair<Point<dim>, Point<dim>>
    get_domain_boundaries() const final
    {
      return {};
    }

    double
    get_r_max() const final
    {
      return 1.;
    }

    unsigned int
    n_order_parameters() const final
    {
      return 2;
    }

    unsigned int
    n_particles() const final
    {
      return 2;
    }
  };
} // namespace Sintering