// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <pf-applications/grain_tracker/ellipsoid.h>
#include <pf-applications/grain_tracker/representation.h>

#include <memory>

namespace GrainTracker
{
  using namespace dealii;

  template <int dim, typename VectorIds>
  std::vector<double>
  scale_elliptical_representations(
    const std::vector<std::unique_ptr<RepresentationElliptical<dim>>>
                          &representations,
    const DoFHandler<dim> &dof_handler,
    const VectorIds       &particle_ids,
    const double           invalid_particle_id = -1.0)
  {
    const auto comm = dof_handler.get_mpi_communicator();

    const unsigned int n_particles = representations.size();

    // Compute particles moments of inertia
    std::vector<double> particle_scales(n_particles, 1.);
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto unique_id = particle_ids[cell->global_active_cell_index()];

          if (unique_id == invalid_particle_id)
            continue;

          AssertIndexRange(unique_id, n_particles);

          const auto &rep = representations[unique_id];
          const auto &E   = rep->ellipsoid;

          auto dist_vec = cell->barycenter() - rep->ellipsoid.get_center();
          dist_vec += (dist_vec / dist_vec.norm()) * cell->diameter() / 2.;
          const auto p = E.get_center() + dist_vec;

          if (!rep->ellipsoid.point_inside(p))
            {
              const auto [t_inter, overlap] =
                find_ellipsoid_intersection(E, E.get_center(), p);

              if (t_inter > 0 && t_inter < 1)
                particle_scales[unique_id] =
                  std::max(1. / t_inter, particle_scales[unique_id]);
            }
        }

    // Reduce information - maximum particle scales
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_scales.data(),
                  particle_scales.size(),
                  MPI_DOUBLE,
                  MPI_MAX,
                  comm);

    return particle_scales;
  }
} // namespace GrainTracker