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

// Compare behavior for spherical and elliptical representations in GT

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/sintering/initial_values_cloud.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>

#include <iostream>

using namespace dealii;

using Number     = double;
using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution() = default;

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    // One grain circular whereas another is elliptical
    const auto is_ellipse =
      (std::pow(p[0] - 4, 2) / 9. + std::pow(p[1] - 2, 2) / 1.) <= 1.;
    const auto is_circle = p.distance(Point<dim>(4., 6.5)) < 1.5;

    return (is_ellipse || is_circle) ? 1. : 0.;
  }
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim = 2;

  const unsigned int fe_degree = 1;

  const bool is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream                        pcout(std::cout, is_zero_rank);
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  FE_Q<dim>                                 fe(fe_degree);
  MappingQ<dim>                             mapping(1);
  QGauss<dim>                               quad(fe_degree + 1);
  DoFHandler<dim>                           dof_handler(tria);
  AffineConstraints<Number>                 constraint;

  const double width = 9.;

  Point<dim> bottom_left;
  Point<dim> top_right;
  for (unsigned int d = 0; d < dim; ++d)
    top_right[d] = width;
  std::vector<unsigned int> subdivisions(dim, 10);

  // Mesh settings
  const bool         periodic      = false;
  const unsigned int n_refinements = 3;

  Sintering::create_mesh_from_divisions(
    tria, bottom_left, top_right, subdivisions, periodic, n_refinements);

  // setup DoFHandlers
  dof_handler.distribute_dofs(fe);

  // setup constraints
  constraint.clear();
  constraint.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);
  constraint.close();

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler),
    dof_handler.get_communicator());

  // set initial condition
  VectorType solution(1);
  solution.block(0).reinit(partitioner);
  solution.zero_out_ghost_values();

  VectorTools::interpolate(mapping,
                           dof_handler,
                           Solution<dim>(),
                           solution.block(0));
  constraint.distribute(solution.block(0));

  solution.update_ghost_values();

  pcout << std::boolalpha;

  // Grain tracker settings
  const double       threshold_lower          = 1e-15;
  const double       threshold_new_grains     = 1e-15;
  const double       buffer_distance_ratio    = 0.05;
  const double       buffer_distance_fixed    = 0.0;
  const bool         allow_new_grains         = false;
  const bool         fast_reassignment        = false;
  const bool         greedy_init              = false;
  const unsigned int op_offset                = 0;
  const unsigned int max_order_parameters_num = 2;

  // Grain tracker with spherical grains representation
  GrainTracker::Tracker<dim, Number> grain_tracker_spherical(
    dof_handler,
    tria,
    greedy_init,
    allow_new_grains,
    fast_reassignment,
    /* elliptical_grains = */ false,
    max_order_parameters_num,
    threshold_lower,
    threshold_new_grains,
    buffer_distance_ratio,
    buffer_distance_fixed,
    op_offset);

  const auto [n_collisions_sp, grains_reassigned_sp, op_number_changed_sp] =
    grain_tracker_spherical.initial_setup(solution, 1);

  pcout << "Spherical grain representation:" << std::endl;
  pcout << "n_collisions      = " << n_collisions_sp << std::endl;
  pcout << "grains_reassigned = " << grains_reassigned_sp << std::endl;
  pcout << "op_number_changed = " << op_number_changed_sp << std::endl;
  grain_tracker_spherical.print_current_grains(pcout, true);
  pcout << std::endl;

  // Grain tracker with elliptical grains representation
  GrainTracker::Tracker<dim, Number> grain_tracker_elliptical(
    dof_handler,
    tria,
    greedy_init,
    allow_new_grains,
    fast_reassignment,
    /* elliptical_grains = */ true,
    max_order_parameters_num,
    threshold_lower,
    threshold_new_grains,
    buffer_distance_ratio,
    buffer_distance_fixed,
    op_offset);

  const auto [n_collisions_el, grains_reassigned_el, op_number_changed_el] =
    grain_tracker_elliptical.initial_setup(solution, 1);

  pcout << "Elliptical grain representation:" << std::endl;
  pcout << "n_collisions      = " << n_collisions_el << std::endl;
  pcout << "grains_reassigned = " << grains_reassigned_el << std::endl;
  pcout << "op_number_changed = " << op_number_changed_el << std::endl;

  grain_tracker_elliptical.print_current_grains(pcout, true);

  solution.zero_out_ghost_values();
}
