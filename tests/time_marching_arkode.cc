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

#include <deal.II/base/mpi.h>

#include <pf-applications/tests/time_marching_tester.h>
#include <pf-applications/time_integration/time_integrator_data.h>
#include <pf-applications/time_integration/time_marching.h>

using namespace dealii;
using namespace TimeIntegration;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Test::TimeMarchingTester tester;

  // Create time marching for each scheme with the corresponding state vector
  std::vector<Test::MarchingSet> time_marchings;

  const auto dt =
    tester.nonlinear_operator.get_data().time_data.get_current_dt();

  // Add Arkode time marching
  TimeIntegrationData time_integration_data;
  time_integration_data.time_start    = 0;
  time_integration_data.time_end      = dt * tester.n_steps;
  time_integration_data.time_step_min = dt;

  auto arkode_factory = [&](NonLinearSolvers::NewtonSolverSolverControl &stats,
                            MyTimerOutput &timer) {
    return std::make_unique<
      TimeMarchingArkode<Test::BlockVectorType, Test::SinOperator>>(
      tester.nonlinear_operator,
      tester.residual_wrapper,
      tester.constraints,
      time_integration_data,
      stats,
      timer);
  };

  // Default Arkode settings
  time_marchings.emplace_back("Arkode default",
                              tester.solution,
                              tester.nonlinear_data,
                              tester.timer,
                              arkode_factory);

  // FE method and disable adaptive stepping

  /* Disable this test, since old Sundials version is in the docker image
  time_integration_data.growth_factor = 1.0;
  time_integration_data.arkode_data.explicit_method_name =
    "ARKODE_FORWARD_EULER_1_1";
  time_integration_data.arkode_data.implicit_method_name = "ARKODE_DIRK_NONE";
  time_integration_data.arkode_data.maximum_order        = 0;

  time_marchings.emplace_back("Arkode FE",
                              tester.solution,
                              tester.nonlinear_data,
                              tester.timer,
                              arkode_factory);
  */

  tester.test(time_marchings, false);

  return 0;
}
