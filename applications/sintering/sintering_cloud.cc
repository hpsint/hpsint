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

// Sintering of N particles loaded from a CSV file

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
#include <pf-applications/sintering/initial_values_cloud.h>

#include <cstdlib>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;
  std::string           file_cloud;

  if (argc == 2)
    {
      if (std::string(argv[1]) == "--help")
        {
          dealii::ConditionalOStream pcout(
            std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

          pcout << "Arguments list: cloud_file [param_file]" << std::endl;
          pcout
            << "    cloud_file - a path to a CSV file containing particles list"
            << std::endl;
          pcout << "    param_file - a path to a prm file" << std::endl;

          pcout << std::endl;
          params.print();
          return 0;
        }
      else
        {
          file_cloud = std::string(argv[1]);
        }
    }
  else if (argc == 3)
    {
      file_cloud = std::string(argv[1]);
      params.parse(std::string(argv[2]));
    }
  else
    {
      AssertThrow(false, ExcMessage("Argument cloud_file has to be provided"));
    }

  std::ifstream fstream(file_cloud.c_str());

  const auto particles = Sintering::read_particles<SINTERING_DIM>(fstream);

  const auto initial_solution =
    std::make_shared<Sintering::InitialValuesCloud<SINTERING_DIM>>(
      particles,
      params.geometry_data.interface_width,
      params.geometry_data.minimize_order_parameters,
      params.geometry_data.interface_buffer_ratio);

  AssertThrow(initial_solution->n_order_parameters() <= MAX_SINTERING_GRAINS,
              Sintering::ExcMaxGrainsExceeded(
                initial_solution->n_order_parameters(), MAX_SINTERING_GRAINS));

  Sintering::Problem<SINTERING_DIM> runner(params, initial_solution);
  runner.run();
}
