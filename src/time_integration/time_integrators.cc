// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <pf-applications/time_integration/time_integrators.h>

unsigned int
TimeIntegration::get_scheme_order(std::string scheme)
{
  unsigned int time_integration_order = 0;
  if (scheme == "BDF1")
    time_integration_order = 1;
  else if (scheme == "BDF2")
    time_integration_order = 2;
  else if (scheme == "BDF3")
    time_integration_order = 3;
  else
    AssertThrow(false, ExcNotImplemented());

  return time_integration_order;
}
