// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  AssertDimension(n_procs, 4);
  (void)n_procs;

  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input;

  if (my_rank == 0)
    {
      input.resize(1);
      input[0].emplace_back(0, 1);
      input[0].emplace_back(0, 4);
    }
  else if (my_rank == 1)
    {
      input.resize(3);
      input[0].emplace_back(0, 7);
      input[1].emplace_back(0, 5);
      input[2].emplace_back(0, 8);
    }
  else if (my_rank == 2)
    {
      input.resize(1);
      input[0].emplace_back(0, 0);
      input[0].emplace_back(0, 7);
    }
  else if (my_rank == 3)
    {
      input.resize(4);
      input[0].emplace_back(0, 2);
      input[2].emplace_back(0, 4);
      input[3].emplace_back(0, 3);
    }

  const auto results = Utilities::MPI::gather(
    comm,
    GrainTracker::perform_distributed_stitching_via_graph(comm, input),
    0);

  if (my_rank == 0)
    for (const auto &result : results)
      {
        for (const auto i : result)
          std::cout << i << " ";
        std::cout << std::endl;
      }
}
