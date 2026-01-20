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
#include <pf-applications/time_integration/time_marching.h>

using namespace dealii;
using namespace TimeIntegration;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Test::TimeMarchingTester tester;

  // Time schemes to test
  std::map<std::string, std::unique_ptr<ExplicitScheme>> schemes;
  schemes.emplace("FE", std::make_unique<ForwardEulerScheme>());
  schemes.emplace("RK4", std::make_unique<RungeKutta4Scheme>());

  // Create time marching for each scheme with the corresponding state vector
  std::vector<Test::MarchingSet> time_marchings;

  for (auto &[label, scheme] : schemes)
    {
      auto explicit_factory =
        [&](NonLinearSolvers::NewtonSolverSolverControl &stats,
            MyTimerOutput                               &timer) {
          return std::make_unique<TimeMarchingExplicit<Test::BlockVectorType,
                                                       Test::SinOperator,
                                                       std::ostream>>(
            tester.nonlinear_operator,
            tester.residual_wrapper,
            tester.constraints,
            std::move(scheme),
            stats,
            timer,
            std::cout);
        };

      time_marchings.emplace_back(label,
                                  tester.solution,
                                  tester.nonlinear_data,
                                  tester.timer,
                                  explicit_factory);
    }

  tester.test(time_marchings);

  return 0;
}
