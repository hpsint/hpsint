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

// Particles of different order parameters that touch each other only across a
// periodic boundary are direct neighbors if the mesh is periodic.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>

#include <iostream>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

constexpr int    dim                 = 2;
constexpr double invalid_particle_id = -1.0;

void
run(const bool periodic, ConditionalOStream &pcout)
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(
    tria, {2, 2}, Point<dim>(0., 0.), Point<dim>(8., 8.), true);

  if (periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodicity_vector;
      GridTools::collect_periodic_faces(tria, 0, 1, 0, periodicity_vector);
      tria.add_periodicity(periodicity_vector);
    }

  tria.refine_global(3);

  DoFHandler<dim> dof_handler(tria);

  /* The order parameter 0 holds the column of cells at the left boundary and
   * the order parameter 1 holds the column at the right boundary. */
  LinearAlgebra::distributed::DynamicBlockVector<double> particle_ids(2);
  for (unsigned int b = 0; b < 2; ++b)
    {
      particle_ids.block(b).reinit(
        tria.global_active_cell_index_partitioner().lock());
      particle_ids.block(b) = invalid_particle_id;
    }

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto x = cell->center()[0];

        if (x < 1.)
          particle_ids.block(0)[cell->global_active_cell_index()] = 0;
        else if (x > 7.)
          particle_ids.block(1)[cell->global_active_cell_index()] = 0;
      }

  for (unsigned int b = 0; b < 2; ++b)
    particle_ids.block(b).update_ghost_values();

  const auto neighbors =
    get_direct_neighbors(dof_handler, particle_ids, invalid_particle_id);

  pcout << "periodic mesh = " << std::boolalpha << periodic << std::endl;
  if (neighbors.empty())
    pcout << "  no direct neighbors" << std::endl;
  for (const auto &[primary, secondaries] : neighbors)
    for (const auto &secondary : secondaries)
      pcout << "  (" << primary.first << ", " << primary.second << ") -> ("
            << secondary.first << ", " << secondary.second << ")" << std::endl;

  for (unsigned int b = 0; b < 2; ++b)
    particle_ids.block(b).zero_out_ghost_values();
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  run(true, pcout);
  run(false, pcout);
}
