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

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;
using namespace GrainTracker;
using namespace Sintering;

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution() = default;

  // 2 circles and 1 ellipse
  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;

    if (p.distance(Point<dim>(1.0, 0.5)) < 0.4)
      return 1.0;
    if ((std::pow(p[0] - 2., 2) / 0.16 + std::pow(p[1] - 0.5, 2) / 0.04) <= 1.)
      return 1.0;
    if (p.distance(Point<dim>(3.0, 0.75)) < 0.2)
      return 1.0;

    return 0.0;
  }
};

constexpr double invalid_particle_id = -1.0;

using BlockVectorType = LinearAlgebra::distributed::DynamicBlockVector<double>;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  AssertDimension(n_procs, 4);
  (void)n_procs;

  const unsigned int dim = 2;

  const unsigned int fe_degree      = 1;
  const unsigned int n_points_1D    = 2;
  const unsigned int n_subdivisions = 1;

  FE_Q<dim>      fe{fe_degree};
  MappingQ1<dim> mapping;
  Quadrature<1>  quad(QIterated<1>(QGauss<1>(n_points_1D), n_subdivisions));

  parallel::distributed::Triangulation<dim> tria(comm);

  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {4, 1},
                                            {0.0, 0.0},
                                            {4.0, 1.0});

  const unsigned int n_refines_global = 4;
  const unsigned int n_refines_local  = 3;

  const double       top_fraction_of_cells    = 0.9;
  const double       bottom_fraction_of_cells = 0.1;
  const unsigned int max_refinement_depth     = 1;
  const unsigned int min_refinement_depth     = 3;
  const double       interface_val_min        = 0.05;
  const double       interface_val_max        = 0.95;

  tria.refine_global(n_refines_global);

  const unsigned int n_global_levels_0 =
    tria.n_global_levels() + n_refines_local;

  // and limit the number of levels
  const unsigned int max_allowed_level =
    (n_global_levels_0 - 1) + max_refinement_depth;
  const unsigned int min_allowed_level =
    (n_global_levels_0 - 1) -
    std::min((n_global_levels_0 - 1), min_refinement_depth);

  DoFHandler<dim>           dof_handler(tria);
  AffineConstraints<double> constraints;
  BlockVectorType           dbv_wrapper(1);
  auto &                    solution = dbv_wrapper.block(0);

  const auto initialize_dofs = [&]() {
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    constraints.close();

    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());

    solution.reinit(partitioner);

    solution.zero_out_ghost_values();
  };

  Solution<dim> solution_data;

  auto reinit_solution_vector = [&]() {
    initialize_dofs();
    VectorTools::interpolate(mapping, dof_handler, solution_data, solution);
    constraints.distribute(solution);
  };

  reinit_solution_vector();
  solution.update_ghost_values();

  for (unsigned int i = 0; i < n_refines_local; ++i)
    {
      coarsen_and_refine_mesh(dbv_wrapper,
                              tria,
                              dof_handler,
                              Quadrature<dim - 1>(quad),
                              top_fraction_of_cells,
                              bottom_fraction_of_cells,
                              min_allowed_level,
                              max_allowed_level,
                              interface_val_min,
                              interface_val_max);

      reinit_solution_vector();
      solution.update_ghost_values();
    }

  LinearAlgebra::distributed::Vector<double> particle_ids(
    tria.global_active_cell_index_partitioner().lock());
  particle_ids = invalid_particle_id;

  const bool   stitching_via_graphs = false;
  const double threshold            = 1e-9;

  // Run flooding and determine local particles, give them local ids and stitch
  // these local numbers
  const auto [offset, local_to_global_particle_ids, local_particle_max_values] =
    detect_local_particle_groups(particle_ids,
                                 dof_handler,
                                 solution,
                                 stitching_via_graphs,
                                 threshold,
                                 invalid_particle_id);

  // Number of particles
  const unsigned int n_particles =
    number_of_stitched_particles(local_to_global_particle_ids, comm);

  // Determine properties of particles (volume, radius, center)
  std::vector<double> particle_info(n_particles * (1 + dim));

  // Compute local information
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        const auto particle_id = particle_ids[cell->global_active_cell_index()];

        if (particle_id == invalid_particle_id)
          continue;

        const unsigned int unique_id =
          local_to_global_particle_ids[static_cast<unsigned int>(particle_id) -
                                       offset];

        AssertIndexRange(unique_id, n_particles);

        particle_info[(dim + 1) * unique_id + 0] += cell->measure();

        for (unsigned int d = 0; d < dim; ++d)
          particle_info[(dim + 1) * unique_id + 1 + d] +=
            cell->center()[d] * cell->measure();
      }

  // Reduce information
  MPI_Reduce(my_rank == 0 ? MPI_IN_PLACE : particle_info.data(),
             particle_info.data(),
             particle_info.size(),
             MPI_DOUBLE,
             MPI_SUM,
             0,
             comm);

  // Output
  Vector<double> ranks(tria.n_active_cells());
  Vector<double> particle_ids_local(tria.n_active_cells());
  ranks = my_rank;

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      particle_ids_local[cell->active_cell_index()] =
        particle_ids[cell->global_active_cell_index()];

  // output particles
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.add_data_vector(ranks,
                           "ranks",
                           DataOut<dim>::DataVectorType::type_cell_data);
  data_out.add_data_vector(particle_ids_local,
                           "particle_ids",
                           DataOut<dim>::DataVectorType::type_cell_data);

  const auto assessment_distances =
    estimate_particle_distances(particle_ids,
                                local_to_global_particle_ids,
                                offset,
                                dof_handler,
                                invalid_particle_id,
                                nullptr,
                                &data_out);

  if (my_rank == 0)
    {
      for (unsigned int i = 0; i < n_particles; ++i)
        {
          std::cout << "Particle " << std::to_string(i) << " has volume "
                    << std::sqrt(particle_info[i * (1 + dim)] / numbers::PI)
                    << " and has a center ("
                    << particle_info[i * (1 + dim) + 1] /
                         particle_info[i * (1 + dim)];
          for (unsigned int d = 1; d < dim; ++d)
            std::cout << ", "
                      << particle_info[i * (1 + dim) + 1 + d] /
                           particle_info[i * (1 + dim)];
          std::cout << ")" << std::endl;
        }

      for (const auto &[key, dist] : assessment_distances)
        std::cout << "distance from " << key.first << " to " << key.second
                  << " = " << dist << std::endl;
    }

  // Generate output
  data_out.build_patches(mapping);
  data_out.write_vtu_in_parallel("solution_distances.vtu", comm);
}
