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

#define SINTERING_DIM 2
#define MAX_SINTERING_GRAINS 2
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

#include <pf-applications/sintering/free_energy.h>
#include <pf-applications/sintering/operator_sintering_generic.h>
#include <pf-applications/sintering/runner.h>

#include <ostream>

using namespace Sintering;

template void
Sintering::runner<SinteringOperatorGeneric, FreeEnergy>(
  int           argc,
  char        **argv,
  std::ostream &out,
  std::ostream &out_statistics);
