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

// Inertia of the particles takes into account the own extent of the cells, so
// that a cluster one cell wide does not degenerate to a zero-width ellipse.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>

#include <pf-applications/base/debug.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>
#include <pf-applications/grain_tracker/representation.h>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace dealii;
using namespace GrainTracker;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int      dim                 = 2;
  constexpr double   invalid_particle_id = -1.0;
  const unsigned int n_particles         = 2;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << std::fixed << std::setprecision(4);

  // 8 x 8 cells of the unit size
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, 0., 8.);
  tria.refine_global(3);

  DoFHandler<dim> dof_handler(tria);

  /* Particle 0 is a column one cell wide, particle 1 is a block of 2 x 2
   * cells. */
  LinearAlgebra::distributed::Vector<double> particle_ids(
    tria.global_active_cell_index_partitioner().lock());
  particle_ids = invalid_particle_id;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto c = cell->center();

        if (c[0] < 1.)
          particle_ids[cell->global_active_cell_index()] = 0;
        else if (c[0] > 5. && c[0] < 7. && c[1] > 2. && c[1] < 4.)
          particle_ids[cell->global_active_cell_index()] = 1;
      }

  const std::vector<Point<dim>> centers  = {Point<dim>(0.5, 4.),
                                            Point<dim>(6., 3.)};
  const std::vector<double>     measures = {8., 4.};

  const auto inertia = compute_particles_inertia(dof_handler,
                                                 particle_ids,
                                                 centers,
                                                 invalid_particle_id);

  for (unsigned int i = 0; i < n_particles; ++i)
    {
      const double *data = &inertia[i * num_inertias<dim>];

      pcout << "particle " << i << std::endl;
      if (pcout.is_active())
        debug::print_components(std::vector<double>(data,
                                                    data + num_inertias<dim>),
                                "  inertia",
                                std::cout);

      const RepresentationElliptical<dim> representation(centers[i],
                                                         measures[i],
                                                         data);
      if (pcout.is_active())
        debug::print_components(representation.ellipsoid.get_radii(),
                                "  radii",
                                std::cout);
    }
}
