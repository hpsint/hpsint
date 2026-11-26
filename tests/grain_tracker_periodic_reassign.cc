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

// Two grains sharing an order parameter are close to each other only across
// the periodic boundary. The grain tracker has to detect the collision if the
// mesh is periodic and ignore it otherwise.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <pf-applications/tests/periodic_grains_tester.h>

#include <iomanip>
#include <iostream>
#include <set>

using namespace dealii;
using namespace PeriodicGrainsTester;

void
run(const bool periodic)
{
  const bool is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream pcout(std::cout, is_zero_rank);

  Domain domain(periodic);

  /* The grains span x = 0.5 .. 3.5 and x = 6.5 .. 9.5. The distance between
   * them is 3 inside the domain and 1 across the periodic boundary. */
  const std::vector<Disc> discs = {{Point<dim>(2., 5.), 1.5, 0},
                                   {Point<dim>(8., 5.), 1.5, 0}};

  const unsigned int n_order_parameters = 2;

  auto solution = domain.make_solution(discs, n_order_parameters);
  solution.update_ghost_values();

  TrackerSettings settings;
  settings.buffer_distance_fixed = 1.5;

  auto tracker = make_tracker(domain, settings);

  const auto [n_collisions, grains_reassigned, op_number_changed] =
    tracker.initial_setup(solution, n_order_parameters);

  std::set<unsigned int> order_parameters;
  for (const auto &[gid, grain] : tracker.get_grains())
    {
      (void)gid;
      order_parameters.insert(grain.get_order_parameter_id());
    }

  pcout << std::boolalpha;
  pcout << "periodic mesh              = " << periodic << std::endl;
  pcout << "n_grains                   = " << tracker.get_grains().size()
        << std::endl;
  pcout << "n_collisions               = " << n_collisions << std::endl;
  pcout << "grains_reassigned          = " << grains_reassigned << std::endl;
  pcout << "op_number_changed          = " << op_number_changed << std::endl;
  pcout << "n_order_parameters_in_use  = " << order_parameters.size()
        << std::endl;

  pcout << std::fixed << std::setprecision(2);
  for (const auto *grain : sorted_grains(tracker))
    pcout << "grain at x = " << grain->get_segments()[0].get_center()[0]
          << ", segments = " << grain->n_segments() << std::endl;
  pcout << std::endl;

  solution.zero_out_ghost_values();
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  run(true);
  run(false);
}
