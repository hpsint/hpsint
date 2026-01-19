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

#include <string>

namespace TimeIntegration
{
  // The default values correspond to the default ARKODE settings
  struct ArkodeData
  {
    unsigned int maximum_order           = 4;
    std::string  explicit_method_name    = "";
    std::string  implicit_method_name    = "";
    bool         time_step_size_adaptive = true;
  };
} // namespace TimeIntegration
