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

#pragma once

#include <pf-applications/time_integration/arkode_parameters.h>

#include <string>

namespace TimeIntegration
{
  struct TimeIntegrationData
  {
    std::string integration_scheme = "BDF2";
    std::string predictor          = "None";

    double       time_start                  = 0;
    double       time_end                    = 1e3;
    double       time_step_init              = 1e-3;
    double       time_step_min               = 1e-5;
    double       time_step_max               = 1e2;
    double       growth_factor               = 1.2;
    unsigned int desirable_newton_iterations = 5;
    unsigned int desirable_linear_iterations = 100;
    bool         sanity_check_predictor      = false;
    bool         sanity_check_solution       = false;

    ArkodeData arkode_data;
  };
} // namespace TimeIntegration
