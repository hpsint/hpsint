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

// Sintering of 2 particles

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

//#define WITH_TIMING
//#define WITH_TRACKER

#include "sintering/sintering_impl.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;

  if (argc == 2)
    {
      if (std::string(argv[1]) == "--help")
        {
          params.print();
          return 0;
        }
      else
        {
          params.parse(std::string(argv[1]));
        }
    }

  Sintering::Problem<SINTERING_DIM, SINTERING_GRAINS> runner(params);
  runner.run();
}
