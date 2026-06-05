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

namespace Sintering
{
  struct AMGData
  {
    unsigned int smoother_sweeps = 4;
    unsigned int n_cycles        = 5;
  };

  struct ILUData
  {
    unsigned int ilu_fill = 0;
    double       ilu_atol = 0.0;
    double       ilu_rtol = 1.0;
    unsigned int overlap  = 0;
  };

  struct ICData
  {
    unsigned int ic_fill = 0;
    double       ic_atol = 0.0;
    double       ic_rtol = 1.0;
    unsigned int overlap = 0;
  };

  struct PreconditionerData
  {
    PreconditionerData() = default;
    PreconditionerData(const std::string &type)
      : type(type)
    {}

    std::string type = "ILU";
    AMGData     amg_data;
    ILUData     ilu_data;
    ICData      ic_data;
  };

  struct BlockPreconditioner2Data
  {
    PreconditionerData block_0 = {"ILU"};
    PreconditionerData block_1 = {"InverseDiagonalMatrix"};
    PreconditionerData block_2 = {"AMG"};

    std::string block_1_approximation = "all";
  };
} // namespace Sintering
