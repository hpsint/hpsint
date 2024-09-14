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

  const double width = 10.;

  Point<dim> bottom_left;
  Point<dim> top_right;
  for (unsigned int d = 0; d < dim; ++d)
    {
      top_right[d] = width;
    }
  std::vector<unsigned int> subdivisions(dim, 10);

  // Mesh settings
  const double       interface_width = 1.0;
  const bool         periodic        = true;
  const unsigned int n_refinements   = 3;


  Sintering::create_mesh_from_divisions(
    tria, bottom_left, top_right, subdivisions, periodic, n_refinements);

  // setup DoFHandlers
  dof_handler.distribute_dofs(fe);

  // setup constraints
  constraint.clear();
  constraint.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);

  // add periodic
  if (periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(
            dof_handler, 2 * d, 2 * d + 1, d, periodicity_vector);
        }

      DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                       constraint);
    }

  constraint.close();

  // Read particles
  std::stringstream iss;
  iss << "#x,y,z,r" << std::endl;
  iss << "0.0,0.0,0.0,1.5" << std::endl;
  iss << "10.0,0.0,0.0,1.5" << std::endl;
  iss << "0.0,10.0,0.0,1.5" << std::endl;
  iss << "10.0,10.0,0.0,1.5" << std::endl;
  iss << "5.0,5.0,0.0,1.5" << std::endl;

  const auto   particles                 = Sintering::read_particles<dim>(iss);
  const bool   minimize_order_parameters = true;
  const double interface_buffer_ratio    = 0.5;

  Sintering::InitialValuesCloud<dim> initial_solution(particles,
                                                      interface_width,
                                                      minimize_order_parameters,
                                                      interface_buffer_ratio);

  // set initial condition
  VectorType solution(initial_solution.n_components());

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler),
    dof_handler.get_communicator());

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      solution.block(c).reinit(partitioner);
    }
  solution.zero_out_ghost_values();

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      initial_solution.set_component(c);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               initial_solution,
                               solution.block(c));

      constraint.distribute(solution.block(c));
    }

  // Grain tracker settings
  const double       threshold_lower          = 1e-15;
  const double       threshold_new_grains     = 1e-15;
  const double       buffer_distance_ratio    = 0.05;
  const double       buffer_distance_fixed    = 0.0;
  const bool         allow_new_grains         = false;
  const bool         fast_reassignment        = false;
  const bool         elliptical_grains        = false;
  const bool         greedy_init              = !minimize_order_parameters;
  const unsigned int op_offset                = 2;
  const unsigned int max_order_parameters_num = 5;

  GrainTracker::Tracker<dim, Number> grain_tracker(dof_handler,
                                                   tria,
                                                   greedy_init,
                                                   allow_new_grains,
                                                   fast_reassignment,
                                                   elliptical_grains,
                                                   max_order_parameters_num,
                                                   threshold_lower,
                                                   threshold_new_grains,
                                                   buffer_distance_ratio,
                                                   buffer_distance_fixed,
                                                   op_offset);

  solution.update_ghost_values();
  grain_tracker.initial_setup(solution, initial_solution.n_order_parameters());
  grain_tracker.print_current_grains(pcout, true);
  solution.zero_out_ghost_values();
}
