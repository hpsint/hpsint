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

// The wavefronts grown from the particles pass through the periodic boundary,
// so that the distance between two particles is measured across it if this is
// the shortest way.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

constexpr int    dim                 = 2;
constexpr double invalid_particle_id = -1.0;

void
run(const bool periodic, ConditionalOStream &pcout)
{
  // 16 x 2 cells of the unit size
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_rectangle(
    tria, {16, 2}, Point<dim>(0., 0.), Point<dim>(16., 2.), true);

  if (periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodicity_vector;
      GridTools::collect_periodic_faces(tria, 0, 1, 0, periodicity_vector);
      tria.add_periodicity(periodicity_vector);
    }

  DoFHandler<dim> dof_handler(tria);

  /* Particle 0 occupies columns 0-1 and particle 1 columns 12-13. They are 10
   * cells apart inside the domain and 2 cells apart across the boundary. */
  LinearAlgebra::distributed::DynamicBlockVector<double> particle_ids(1);
  particle_ids.block(0).reinit(
    tria.global_active_cell_index_partitioner().lock());
  particle_ids.block(0) = invalid_particle_id;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto x     = cell->center()[0];
        const auto index = cell->global_active_cell_index();

        if (x < 2.)
          particle_ids.block(0)[index] = 0;
        else if (x > 12. && x < 14.)
          particle_ids.block(0)[index] = 1;
      }

  const auto distances = estimate_particle_distances(particle_ids.block(0),
                                                     dof_handler,
                                                     invalid_particle_id);

  pcout << "periodic mesh = " << std::boolalpha << periodic << std::endl;
  for (const auto &[key, dist] : distances)
    pcout << "  distance from " << key.first << " to " << key.second << " = "
          << dist << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << std::fixed << std::setprecision(4);

  run(true, pcout);
  run(false, pcout);
}
