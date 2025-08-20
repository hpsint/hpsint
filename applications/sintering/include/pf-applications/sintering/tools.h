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

#pragma once

namespace Sintering
{
  struct EnergyCoefficients
  {
    double A;
    double B;
    double kappa_c;
    double kappa_p;
  };

  EnergyCoefficients
  compute_energy_params(const double surface_energy,
                        const double gb_energy,
                        const double interface_width,
                        const double length_scale,
                        const double energy_scale);
} // namespace Sintering