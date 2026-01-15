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

  struct BlockPreconditioner2Data
  {
    std::string block_0_preconditioner = "ILU";
    std::string block_1_preconditioner = "InverseDiagonalMatrix";
    std::string block_2_preconditioner = "AMG";

    std::string block_1_approximation = "all";

    AMGData block_2_amg_data;
  };
} // namespace Sintering
