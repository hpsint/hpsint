// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Sintering of N particles located along the circle

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

//#define WITH_TIMING
//#define WITH_TRACKER

#include <pf-applications/sintering/driver.h>
#include <pf-applications/sintering/initial_values_circle.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;

  if (argc == 2 && std::string(argv[1]) == "--help")
    {
      params.print();
      return 0;
    }

  AssertThrow(2 <= argc && argc <= 3, ExcNotImplemented());

  const unsigned int n_grains = atoi(argv[1]);

  if (argc == 3)
    params.parse(std::string(argv[2]));

  // geometry
  static constexpr double r0              = 15.0 / 2.;
  static constexpr bool   is_accumulative = false;

  const auto initial_solution =
    std::make_shared<Sintering::InitialValuesCircle<SINTERING_DIM>>(
      r0,
      params.geometry_data.interface_width,
      n_grains,
      params.geometry_data.minimize_order_parameters,
      is_accumulative);

  AssertThrow(initial_solution->n_order_parameters() <= MAX_SINTERING_GRAINS,
              Sintering::ExcMaxGrainsExceeded(
                initial_solution->n_order_parameters(), MAX_SINTERING_GRAINS));

  Sintering::Problem<SINTERING_DIM> runner(params, initial_solution);
  runner.run();
}
