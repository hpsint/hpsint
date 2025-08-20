// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <pf-applications/sintering/tools.h>

Sintering::EnergyCoefficients
Sintering::compute_energy_params(const double surface_energy,
                                 const double gb_energy,
                                 const double interface_width,
                                 const double length_scale,
                                 const double energy_scale)
{
  const double scaled_gb_energy =
    gb_energy / energy_scale * length_scale * length_scale;

  const double scaled_surface_energy =
    surface_energy / energy_scale * length_scale * length_scale;

  const double kappa_c = 3.0 / 4.0 *
                         (2.0 * scaled_surface_energy - scaled_gb_energy) *
                         interface_width;
  const double kappa_p = 3.0 / 4.0 * scaled_gb_energy * interface_width;

  const double A =
    (12.0 * scaled_surface_energy - 7.0 * scaled_gb_energy) / interface_width;
  const double B = scaled_gb_energy / interface_width;

  EnergyCoefficients params{A, B, kappa_c, kappa_p};

  return params;
}
