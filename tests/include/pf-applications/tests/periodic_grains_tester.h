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

#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/grid_tools.h>

#include <memory>
#include <vector>

namespace PeriodicGrainsTester
{
  using namespace dealii;

  constexpr int dim = 2;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  // Circular grain living in a given order parameter
  struct Disc
  {
    Point<dim>   center;
    double       radius;
    unsigned int order_parameter;
  };

  /* Order parameters made of discs. Distances are measured to the closest
   * periodic image if the domain is periodic, so the discs may cross the
   * domain boundary. */
  class DiscsFunction : public Function<dim>
  {
  public:
    DiscsFunction(const std::vector<Disc> &discs,
                  const double             width,
                  const bool               periodic)
      : discs(discs)
      , width(width)
      , periodic(periodic)
    {}

    void
    set_component(const unsigned int component_in)
    {
      component = component_in;
    }

    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      for (const auto &disc : discs)
        if (disc.order_parameter == component)
          {
            double dist2 = 0;
            for (unsigned int d = 0; d < dim; ++d)
              {
                double delta = std::abs(p[d] - disc.center[d]);
                if (periodic)
                  delta = std::min(delta, width - delta);
                dist2 += delta * delta;
              }

            if (std::sqrt(dist2) < disc.radius)
              return 1.;
          }

      return 0.;
    }

  private:
    const std::vector<Disc> &discs;
    const double             width;
    const bool               periodic;

    unsigned int component = 0;
  };

  // Square domain with an optionally periodic mesh
  class Domain
  {
  public:
    Domain(const bool periodic, const double width = 10.)
      : periodic(periodic)
      , width(width)
      , tria(MPI_COMM_WORLD)
      , fe(1)
      , mapping(1)
      , dof_handler(tria)
    {
      Point<dim> bottom_left;
      Point<dim> top_right;
      for (unsigned int d = 0; d < dim; ++d)
        top_right[d] = width;

      const std::vector<unsigned int> subdivisions(dim, 10);
      const unsigned int              n_refinements = 3;

      hpsint::create_mesh_from_divisions(tria,
                                         bottom_left,
                                         top_right,
                                         subdivisions,
                                         periodic,
                                         n_refinements,
                                         false);

      dof_handler.distribute_dofs(fe);

      constraints.clear();
      constraints.reinit(dof_handler.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      if (periodic)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>>
            periodicity_vector;

          for (unsigned int d = 0; d < dim; ++d)
            GridTools::collect_periodic_faces(
              dof_handler, 2 * d, 2 * d + 1, d, periodicity_vector);

          DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                           constraints);
        }

      constraints.close();

      partitioner = std::make_shared<Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(dof_handler),
        dof_handler.get_mpi_communicator());
    }

    // Solution without ghost values updated
    VectorType
    make_solution(const std::vector<Disc> &discs,
                  const unsigned int       n_order_parameters) const
    {
      VectorType solution(n_order_parameters);

      DiscsFunction function(discs, width, periodic);

      for (unsigned int c = 0; c < n_order_parameters; ++c)
        {
          solution.block(c).reinit(partitioner);
          solution.block(c).zero_out_ghost_values();

          function.set_component(c);
          VectorTools::interpolate(mapping,
                                   dof_handler,
                                   function,
                                   solution.block(c));

          constraints.distribute(solution.block(c));
        }

      return solution;
    }

    const bool   periodic;
    const double width;

    parallel::distributed::Triangulation<dim> tria;
    FE_Q<dim>                                 fe;
    MappingQ<dim>                             mapping;
    DoFHandler<dim>                           dof_handler;
    AffineConstraints<Number>                 constraints;

    std::shared_ptr<Utilities::MPI::Partitioner> partitioner;
  };

  // Settings of the grain tracker, the rest is not important for the tests
  struct TrackerSettings
  {
    bool         greedy_init              = false;
    bool         allow_new_grains         = false;
    bool         fast_reassignment        = false;
    unsigned int max_order_parameters_num = 2;
    double       threshold_lower          = 0.01;
    double       threshold_new_grains     = 0.05;
    double       buffer_distance_ratio    = 0.05;
    double       buffer_distance_fixed    = 0.0;
    unsigned int op_offset                = 0;
  };

  inline GrainTracker::Tracker<dim, Number>
  make_tracker(const Domain &domain, const TrackerSettings &settings)
  {
    return GrainTracker::Tracker<dim, Number>(
      domain.dof_handler,
      domain.tria,
      settings.greedy_init,
      settings.allow_new_grains,
      settings.fast_reassignment,
      settings.max_order_parameters_num,
      GrainTracker::GrainRepresentation::spherical,
      settings.threshold_lower,
      settings.threshold_new_grains,
      settings.buffer_distance_ratio,
      settings.buffer_distance_fixed,
      settings.op_offset);
  }

  // Grains sorted by the x coordinate of their largest segment
  inline std::vector<const GrainTracker::Grain<dim> *>
  sorted_grains(const GrainTracker::Tracker<dim, Number> &tracker)
  {
    std::vector<const GrainTracker::Grain<dim> *> result;
    for (const auto &[gid, grain] : tracker.get_grains())
      {
        (void)gid;
        result.push_back(&grain);
      }

    const auto largest_segment_center_x = [](const auto *grain) {
      const auto &segments = grain->get_segments();
      return std::max_element(segments.begin(),
                              segments.end(),
                              [](const auto &a, const auto &b) {
                                return a.get_measure() < b.get_measure();
                              })
        ->get_center()[0];
    };

    std::sort(result.begin(), result.end(), [&](const auto *a, const auto *b) {
      return largest_segment_center_x(a) < largest_segment_center_x(b);
    });

    return result;
  }
} // namespace PeriodicGrainsTester
