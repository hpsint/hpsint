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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/revision.h>

#include <pf-applications/base/revision.h>

#include <pf-applications/sintering/operator_grand_potential_greenquist.h>
#include <pf-applications/sintering/runner.h>
#include <pf-applications/sintering/tools.h>

using namespace dealii;
using namespace Sintering;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "Running: " << concatenate_strings(argc, argv) << std::endl;
  pcout << "  - with n MPI processes:"
        << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
  pcout << "  - deal.II (branch: " << DEAL_II_GIT_BRANCH
        << "; revision: " << DEAL_II_GIT_REVISION
        << "; short: " << DEAL_II_GIT_SHORTREV << ")" << std::endl;
  pcout << "  - hpsint (branch: " << PF_APPLICATIONS_GIT_BRANCH
        << "; revision: " << PF_APPLICATIONS_GIT_REVISION
        << "; short: " << PF_APPLICATIONS_GIT_SHORTREV << ")" << std::endl;
  pcout << std::endl;
  pcout << std::endl;

  runner<GreenquistGrandPotentialOperator, GreenquistFreeEnergy>(argc, argv);

  return 0;
}
