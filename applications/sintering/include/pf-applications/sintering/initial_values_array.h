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

#pragma once

#include <pf-applications/sintering/initial_values_spherical.h>

namespace Sintering
{
  template <int dim>
  class InitialValuesArray : public InitialValuesSpherical<dim>
  {
  public:
    InitialValuesArray(
      const double             r0,
      const double             interface_width,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset = 2,
      const bool               is_accumulative      = false)
      : InitialValuesSpherical<dim>(interface_width,
                                    interface_direction,
                                    op_components_offset,
                                    is_accumulative)
      , r0(r0)
    {}

    double
    op_value(const dealii::Point<dim> &p,
             const unsigned int        order_parameter) const final
    {
      double ret_val = 0;

      for (const auto pid : order_parameter_to_grains.at(order_parameter))
        {
          const auto &pt = centers.at(pid);
          ret_val        = this->is_in_sphere(p, pt, r0);

          if (ret_val != 0)
            break;
        }

      return ret_val;
    }

    std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const final
    {
      const auto pt_xmax = *std::max_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[0] < b[0];
                                             });

      const auto pt_ymax = *std::max_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[1] < b[1];
                                             });

      const auto pt_xmin = *std::min_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[0] < b[0];
                                             });

      const auto pt_ymin = *std::min_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[1] < b[1];
                                             });

      double xmin = pt_xmin[0] - r0;
      double xmax = pt_xmax[0] + r0;
      double ymin = pt_ymin[1] - r0;
      double ymax = pt_ymax[1] + r0;

      if (dim == 2)
        {
          return std::make_pair(dealii::Point<dim>(xmin, ymin),
                                dealii::Point<dim>(xmax, ymax));
        }
      else if (dim == 3)
        {
          const auto pt_zmax =
            *std::max_element(centers.begin(),
                              centers.end(),
                              [](const auto &a, const auto &b) {
                                return a[2] < b[2];
                              });

          const auto pt_zmin =
            *std::min_element(centers.begin(),
                              centers.end(),
                              [](const auto &a, const auto &b) {
                                return a[2] < b[2];
                              });

          double zmin = pt_zmin[2] - r0;
          double zmax = pt_zmax[2] + r0;
          return std::make_pair(dealii::Point<dim>(xmin, ymin, zmin),
                                dealii::Point<dim>(xmax, ymax, zmax));
        }
    }

    double
    get_r_max() const final
    {
      return r0;
    }

    unsigned int
    n_order_parameters() const final
    {
      return order_parameter_to_grains.size();
    }

    unsigned int
    n_particles() const final
    {
      return centers.size();
    }

  protected:
    // Center coordinates
    std::vector<dealii::Point<dim>> centers;

    // Map order parameters to specific grains
    std::map<unsigned int, std::vector<unsigned int>> order_parameter_to_grains;

  private:
    double r0;
  };
} // namespace Sintering