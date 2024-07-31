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

#include <filesystem>
#include <iostream>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace Test
{
  using namespace dealii;
  using namespace Sintering;

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  void
  mesh_cloud(const std::string &cloud, const InitialMesh initial_mesh)
  {
    constexpr int dim = 3;

    // Approximation settings
    const unsigned int fe_degree      = 1;
    const unsigned int n_points_1D    = 2;
    const unsigned int n_subdivisions = 1;

    // Some default objects
    const bool is_zero_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    ConditionalOStream                        pcout(std::cout, is_zero_rank);
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    FE_Q<dim>                                 fe(fe_degree);
    MappingQ<dim>                             mapping(1);
    Quadrature<1>   quad(QIterated<1>(QGauss<1>(n_points_1D), n_subdivisions));
    DoFHandler<dim> dof_handler(tria);
    AffineConstraints<Number> constraints;

    // Open cloud file
    const std::filesystem::path source_path = XSTRING(SOURCE_CODE_ROOT);
    const std::filesystem::path file_cloud =
      source_path /
      "applications/sintering/sintering_cloud_examples/packings_10k" / cloud;

    std::ifstream fstream(file_cloud);
    AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

    const auto particles = read_particles<dim>(fstream);

    // Initial cloud settings
    const double interface_width           = 10;
    const bool   minimize_order_parameters = true;
    const double interface_buffer_ratio    = 3;
    const double radius_buffer_ratio       = 0;

    const auto initial_solution =
      std::make_shared<InitialValuesCloud<dim>>(particles,
                                                interface_width,
                                                minimize_order_parameters,
                                                interface_buffer_ratio,
                                                radius_buffer_ratio);

    auto boundaries = initial_solution->get_domain_boundaries();

    const double r_max           = initial_solution->get_r_max();
    const double boundary_factor = 0.5;

    for (unsigned int d = 0; d < dim; ++d)
      {
        boundaries.first[d] -= boundary_factor * r_max;
        boundaries.second[d] += boundary_factor * r_max;
      }

    // Meshing settings
    const double        divisions_per_interface            = 1;
    const bool          periodic                           = false;
    const unsigned int  max_prime                          = 20;
    const double        max_level0_divisions_per_interface = 1.0 - 1e-9;
    const InitialRefine global_refine = InitialRefine::None;

    unsigned int n_refinements_remaining = 0;
    if (initial_mesh == InitialMesh::Interface)
      {
        n_refinements_remaining =
          create_mesh_from_interface(tria,
                                     boundaries.first,
                                     boundaries.second,
                                     interface_width,
                                     divisions_per_interface,
                                     periodic,
                                     global_refine,
                                     max_prime,
                                     max_level0_divisions_per_interface,
                                     n_subdivisions);
      }
    else if (initial_mesh == InitialMesh::MaxRadius)
      {
        n_refinements_remaining =
          create_mesh_from_radius(tria,
                                  boundaries.first,
                                  boundaries.second,
                                  interface_width,
                                  divisions_per_interface,
                                  r_max,
                                  periodic,
                                  global_refine,
                                  max_prime,
                                  n_subdivisions);
      }
    else
      AssertThrow(false, ExcNotImplemented());

    pcout << "n_refinements_remaining = " << n_refinements_remaining
          << std::endl;
    pcout << "tria.n_global_levels()  = " << tria.n_global_levels()
          << std::endl;
    pcout << "tria.n_cells(0)         = " << tria.n_cells(0) << std::endl;
    pcout << std::endl;

    const unsigned int n_global_levels_0 =
      tria.n_global_levels() + n_refinements_remaining;


    VectorType solution(initial_solution->n_components());

    const double       top_fraction_of_cells    = 0.3 * 3;
    const double       bottom_fraction_of_cells = 0.1;
    const unsigned int max_refinement_depth     = 1;
    const unsigned int min_refinement_depth     = 3;
    const double       interface_val_min        = 0.05;
    const double       interface_val_max        = 0.95;


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

      for (unsigned int c = 0; c < solution.n_blocks(); ++c)
        solution.block(c).reinit(partitioner);

      solution.zero_out_ghost_values();
    };
    initialize_dofs();
    pcout << "Initial n_dofs = " << dof_handler.n_dofs() << std::endl;

    const auto initialize_solution = [&]() {
      for (unsigned int c = 0; c < initial_solution->n_components(); ++c)
        {
          initial_solution->set_component(c);

          VectorTools::interpolate(mapping,
                                   dof_handler,
                                   *initial_solution,
                                   solution.block(c));

          constraints.distribute(solution.block(c));
        }
    };
    initialize_solution();

    const unsigned int n_init_refinements =
      std::max(std::min(tria.n_global_levels() - 1, min_refinement_depth),
               n_global_levels_0 - tria.n_global_levels() +
                 max_refinement_depth);

    const auto execute_coarsening_and_refinement = [&]() {
      // and limit the number of levels
      const unsigned int max_allowed_level =
        (n_global_levels_0 - 1) + max_refinement_depth;
      const unsigned int min_allowed_level =
        (n_global_levels_0 - 1) -
        std::min((n_global_levels_0 - 1), min_refinement_depth);

      std::function<void(VectorType &)> after_amr = [&](VectorType &) {
        initialize_dofs();
      };

      const unsigned int block_estimate_start = 2;
      const unsigned int block_estimate_end = initial_solution->n_components();
      coarsen_and_refine_mesh(solution,
                              tria,
                              dof_handler,
                              constraints,
                              Quadrature<dim - 1>(quad),
                              top_fraction_of_cells,
                              bottom_fraction_of_cells,
                              min_allowed_level,
                              max_allowed_level,
                              after_amr,
                              interface_val_min,
                              interface_val_max,
                              block_estimate_start,
                              block_estimate_end);
    };

    pcout << "Number of local refinements to be performed: "
          << n_init_refinements << std::endl;

    pcout << "refine | n_dofs | n_active_cells" << std::endl;
    pcout << "--------------------------------" << std::endl;
    for (unsigned int i = 0; i < n_init_refinements; ++i)
      {
        execute_coarsening_and_refinement();
        initialize_solution();
        pcout << std::setw(6) << (i + 1) << " | " << std::setw(6)
              << dof_handler.n_dofs() << " | " << std::setw(14)
              << tria.n_active_cells() << std::endl;
      }
  }
} // namespace Test
