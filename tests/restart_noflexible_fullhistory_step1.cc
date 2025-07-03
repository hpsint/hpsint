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

#include <deal.II/base/mpi.h>

#include <pf-applications/tests/runner_sintering_models.h>

#include <string>
#include <vector>

using namespace dealii;
using namespace Test;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  std::vector<std::string> arguments = {
    "dummy",
    "--restart",
    std::string("../../restart_noflexible_fullhistory_step0.") +
      XSTRING(OUTPUT_FOLDER_SUFFIX) + "/mpirun=2/restart_0",
    "--PrintTimeLoop=false",
    "--TimeIntegration.TimeEnd=2e-1",
    "--TimeIntegration.TimeStepInit=1e-2",
    "--Restart.FlexibleOutput=false",
    "--Restart.FullHistory=true"};

  run_sintering_operator_generic(arguments.begin(), arguments.end());

  return 0;
}
