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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <pf-applications/base/tensor.h>

#include <pf-applications/grid/bounding_box_filter.h>

#include <memory>
#include <vector>

using namespace dealii;

namespace hpsint
{
  template <int dim, typename VectorType>
  bool
  coarsen_triangulation(parallel::distributed::Triangulation<dim> &tria_copy,
                        const DoFHandler<dim> &background_dof_handler,
                        DoFHandler<dim>       &background_dof_handler_coarsened,
                        const VectorType      &vector,
                        VectorType            &vector_coarsened,
                        const unsigned int     n_coarsening_steps)
  {
    if (n_coarsening_steps == 0)
      return false;

    tria_copy.copy_triangulation(background_dof_handler.get_triangulation());
    background_dof_handler_coarsened.reinit(tria_copy);
    background_dof_handler_coarsened.distribute_dofs(
      background_dof_handler.get_fe_collection());

    // 1) copy solution so that it has the right ghosting
    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      background_dof_handler_coarsened.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(background_dof_handler_coarsened),
      background_dof_handler_coarsened.get_communicator());

    vector_coarsened.reinit(vector.n_blocks());

    for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
      {
        vector_coarsened.block(b).reinit(partitioner);
        vector_coarsened.block(b).copy_locally_owned_data_from(vector.block(b));
      }

    vector_coarsened.update_ghost_values();

    for (unsigned int i = 0; i < n_coarsening_steps; ++i)
      {
        // 2) mark cells for refinement
        for (const auto &cell : tria_copy.active_cell_iterators())
          if (cell->is_locally_owned() &&
              (static_cast<unsigned int>(cell->level() + 1) ==
               tria_copy.n_global_levels()))
            cell->set_coarsen_flag();

        // 3) perform interpolation and initialize data structures
        tria_copy.prepare_coarsening_and_refinement();

        parallel::distributed::SolutionTransfer<dim,
                                                typename VectorType::BlockType>
          solution_trans(background_dof_handler_coarsened);

        std::vector<const typename VectorType::BlockType *>
          vector_coarsened_ptr(vector_coarsened.n_blocks());
        for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
          vector_coarsened_ptr[b] = &vector_coarsened.block(b);

        solution_trans.prepare_for_coarsening_and_refinement(
          vector_coarsened_ptr);

        tria_copy.execute_coarsening_and_refinement();

        background_dof_handler_coarsened.distribute_dofs(
          background_dof_handler.get_fe_collection());

        const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          background_dof_handler_coarsened.locally_owned_dofs(),
          DoFTools::extract_locally_relevant_dofs(
            background_dof_handler_coarsened),
          background_dof_handler_coarsened.get_communicator());

        for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
          vector_coarsened.block(b).reinit(partitioner);

        std::vector<typename VectorType::BlockType *> solution_ptr(
          vector_coarsened.n_blocks());
        for (unsigned int b = 0; b < vector_coarsened.n_blocks(); ++b)
          solution_ptr[b] = &vector_coarsened.block(b);

        solution_trans.interpolate(solution_ptr);
        vector_coarsened.update_ghost_values();
      }

    return true;
  }

  template <typename Number, typename VectorType>
  void
  update_selected_ghosts(VectorType                   &vector,
                         const VectorOperation::values operation,
                         Utilities::MPI::Partitioner  &partitioner,
                         std::vector<Number>          &ghosts_values,
                         const IndexSet               &ghost_indices,
                         const IndexSet               &larger_ghost_index_set)
  {
    partitioner.set_ghost_indices(ghost_indices, larger_ghost_index_set);

    std::vector<MPI_Request> requests;

    // From test 7
    std::vector<Number> temp_array(partitioner.n_import_indices());

    partitioner.import_from_ghosted_array_start(operation,
                                                3,
                                                make_array_view(ghosts_values),
                                                make_array_view(temp_array),
                                                requests);

    partitioner.import_from_ghosted_array_finish(
      operation,
      ArrayView<const Number>(temp_array.data(), temp_array.size()),
      ArrayView<Number>(vector.get_values(), partitioner.locally_owned_size()),
      make_array_view(ghosts_values),
      requests);

    vector.update_ghost_values();
  }

  /* Filter out those cells which do not fit the bounding box. Currently it
   * is assumed that the mapping is linear. For practical cases, there is
   * not need to go beyond this case, at least at the moment. */
  template <int dim, typename VectorType>
  void
  filter_mesh_withing_bounding_box(
    const DoFHandler<dim>                        &background_dof_handler,
    VectorType                                   &vector,
    const double                                  iso_level,
    std::shared_ptr<const BoundingBoxFilter<dim>> box_filter,
    const double                                  null_value = 0.)
  {
    AssertThrow(std::abs(iso_level - null_value) >
                  std::numeric_limits<double>::epsilon(),
                ExcMessage("iso_level = " + std::to_string(iso_level) +
                           " and null_value = " + std::to_string(null_value) +
                           " have to be different"));

    const auto &fe = background_dof_handler.get_fe();

    std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());

    const bool has_ghost_elements = vector.has_ghost_elements();

    if (has_ghost_elements == false)
      vector.update_ghost_values();

    const auto partitioner_full = std::make_shared<Utilities::MPI::Partitioner>(
      background_dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(background_dof_handler),
      background_dof_handler.get_mpi_communicator());

    auto partitioner_reduced = std::make_shared<Utilities::MPI::Partitioner>(
      background_dof_handler.locally_owned_dofs(),
      background_dof_handler.get_mpi_communicator());

    using Number = typename VectorType::value_type;

    // Make local constraints
    AffineConstraints<Number> constraints;
    const auto                relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(background_dof_handler);

    constraints.clear();
    constraints.reinit(background_dof_handler.locally_owned_dofs(),
                       relevant_dofs);
    DoFTools::make_hanging_node_constraints(background_dof_handler,
                                            constraints);
    constraints.close();

    // With the first loop we eliminate all cells outside of the scope
    for (const auto &cell : background_dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell->get_dof_indices(dof_indices);

        for (unsigned int b = 0; b < vector.n_blocks(); ++b)
          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
            {
              const auto &point    = cell->vertex(i);
              const auto  position = box_filter->position(point);

              auto &global_dof_value = vector.block(b)[dof_indices[i]];
              if (position == BoundingBoxFilter<dim>::Position::Boundary)
                global_dof_value = std::min(global_dof_value, iso_level);
              else if (position == BoundingBoxFilter<dim>::Position::Outside)
                global_dof_value = null_value;
            }
      }

    // Additional smoothening
    const unsigned int n_levels =
      background_dof_handler.get_triangulation().n_global_levels();
    for (unsigned int ilevel = 0; ilevel < n_levels; ++ilevel)
      {
        std::vector<std::map<unsigned int, std::pair<Number, Number>>>
          new_values(vector.n_blocks());

        for (const auto &cell :
             background_dof_handler.active_cell_iterators_on_level(ilevel))
          {
            // Skip cell if not locally owned or not intersected
            if (!cell->is_locally_owned() || !box_filter->intersects(*cell))
              continue;

            cell->get_dof_indices(dof_indices);

            for (unsigned int b = 0; b < vector.n_blocks(); ++b)
              {
                // Check if there is any point value larger than iso_level
                unsigned int n_larger_than_iso = 0;
                for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                  if (vector.block(b)[dof_indices[i]] > iso_level)
                    ++n_larger_than_iso;

                if (n_larger_than_iso == 0)
                  continue;

                // Iterate over each line of the cell
                for (unsigned int il = 0; il < cell->n_lines(); il++)
                  {
                    // DOFs correspnding to the vertices
                    const auto index0 = cell->line(il)->vertex_dof_index(0, 0);
                    const auto index1 = cell->line(il)->vertex_dof_index(1, 0);

                    // The field values associated with those DOFs
                    const auto val0 = vector.block(b)[index0];
                    const auto val1 = vector.block(b)[index1];

                    // If both points are outside of the bounding box or
                    // their values are below the iso level, then skip them
                    const bool point_outside0 =
                      box_filter->point_outside_or_boundary(
                        cell->line(il)->vertex(0));
                    const bool point_outside1 =
                      box_filter->point_outside_or_boundary(
                        cell->line(il)->vertex(1));

                    const bool filter_out0 =
                      (point_outside0 || val0 < iso_level);
                    const bool filter_out1 =
                      (point_outside1 || val1 < iso_level);

                    if (filter_out0 && filter_out1)
                      continue;

                    const double length = cell->line(il)->diameter();

                    // Check if there are intersections with box planes
                    for (const auto &plane : box_filter->get_planes())
                      {
                        const auto [has_itersection, fac, p] =
                          intersect_line_plane(cell->line(il)->vertex(0),
                                               cell->line(il)->vertex(1),
                                               plane.origin,
                                               plane.normal);

                        if (has_itersection && std::abs(fac) < 1.)
                          {
                            const auto d0 = p - cell->line(il)->vertex(0);
                            const auto d1 = p - cell->line(il)->vertex(1);

                            // If the intersection point is indeed within
                            // the line range
                            if (d0 * d1 < 0)
                              {
                                double       val_max;
                                unsigned int index_min;
                                double       fac_ratio;
                                if (val0 > val1)
                                  {
                                    val_max   = val0;
                                    index_min = index1;
                                    fac_ratio = std::abs(fac);
                                  }
                                else
                                  {
                                    val_max   = val1;
                                    index_min = index0;
                                    fac_ratio = 1. - std::abs(fac);
                                  }

                                const double ref_val = val_max - iso_level;
                                const double iso_pos = fac_ratio * length;
                                const double k       = -ref_val / iso_pos;
                                const double val_min = k * length + val_max;

                                if (std::abs(vector.block(b)[index_min] -
                                             val_min) > 1e-6)
                                  {
                                    // If not an owner modifies the entry,
                                    // then we store an old and a new values
                                    // and then sync them later below
                                    if (partitioner_full->is_ghost_entry(
                                          index_min))
                                      {
                                        if (new_values[b].find(index_min) ==
                                            new_values[b].end())
                                          new_values[b].try_emplace(
                                            index_min,
                                            std::make_pair(vector.block(
                                                             b)[index_min],
                                                           val_min));
                                        else
                                          new_values[b].at(index_min).second =
                                            val_min;
                                      }

                                    vector.block(b)[index_min] = val_min;
                                  }
                              }
                          }
                      }
                  }
              }
          }

        const double eps_tol = 1e-6;

        // Update modified ghosts
        for (unsigned int b = 0; b < vector.n_blocks(); ++b)
          {
            // This will overrite ghost values if any of them was modified
            // not by the owner, this is what we exactly want
            vector.block(b).update_ghost_values();

            IndexSet local_relevant_reduced(partitioner_full->size());
            std::vector<Number> ghosts_values;

            /* 1. Attempt to nullify the owner value.
             *
             * If a dof value was modified as a ghost not by an owner, we
             * then need to transfer this new value to the owner. But that
             * should be done in a complex way, since multiple ranks could
             * have contributed to this new value. None of the default
             * VectorOperation's fit our needs and this justifies the need
             * for the algo below.
             */
            std::vector<unsigned int> indices_to_remove;
            for (const auto &[index, value] : new_values[b])
              if (std::abs(vector.block(b)[index] - value.first) < eps_tol)
                {
                  local_relevant_reduced.add_index(index);
                  ghosts_values.push_back(-value.first);
                }
              else
                {
                  // We get here if a dof value was modified by the owner,
                  // then we neglect the modifications made by other ranks
                  indices_to_remove.push_back(index);
                }

            for (const auto &index : indices_to_remove)
              new_values[b].erase(index);

            update_selected_ghosts(vector.block(b),
                                   VectorOperation::add,
                                   *partitioner_reduced,
                                   ghosts_values,
                                   local_relevant_reduced,
                                   partitioner_full->ghost_indices());

            /* 2. Nullify any negative owner value if needed
             *
             * If a dof, that had initial value val0, was modified by not a
             * single but K ranks, than after the first step it won't get
             * nullified, but rather will be equal to -(K-1)*val0. We then
             * nullify it using operation max(0, -(K-1)*val0).
             */
            local_relevant_reduced.clear();
            ghosts_values.clear();
            for (const auto &[index, value] : new_values[b])
              if (vector.block(b)[index] < -eps_tol)
                {
                  local_relevant_reduced.add_index(index);
                  ghosts_values.push_back(0.);
                }

            update_selected_ghosts(vector.block(b),
                                   VectorOperation::max,
                                   *partitioner_reduced,
                                   ghosts_values,
                                   local_relevant_reduced,
                                   partitioner_full->ghost_indices());

            /* 3. Set up negative values
             *
             * After the first two steps the dof value, that was modified on
             * any non-owner and not touched on the owner, is guaranteed to
             * be 0 on the owner. We then apply min operation to set up
             * those new values which are negative.
             */
            local_relevant_reduced.clear();
            ghosts_values.clear();
            for (const auto &[index, value] : new_values[b])
              if (value.second < 0)
                {
                  local_relevant_reduced.add_index(index);
                  ghosts_values.push_back(value.second);
                }

            update_selected_ghosts(vector.block(b),
                                   VectorOperation::min,
                                   *partitioner_reduced,
                                   ghosts_values,
                                   local_relevant_reduced,
                                   partitioner_full->ghost_indices());

            /* 4. Set up positive values
             *
             * This step does the same as step 3 but for those new values
             * which are positive.
             */
            local_relevant_reduced.clear();
            ghosts_values.clear();
            for (const auto &[index, value] : new_values[b])
              if (value.second > 0)
                {
                  local_relevant_reduced.add_index(index);
                  ghosts_values.push_back(value.second);
                }

            update_selected_ghosts(vector.block(b),
                                   VectorOperation::max,
                                   *partitioner_reduced,
                                   ghosts_values,
                                   local_relevant_reduced,
                                   partitioner_full->ghost_indices());

            if (ilevel < n_levels - 1)
              {
                vector.block(b).zero_out_ghost_values();
                constraints.distribute(vector.block(b));
                vector.block(b).update_ghost_values();
              }
          }
      }

    if (has_ghost_elements == false)
      vector.zero_out_ghost_values();
  }

  enum class InitialRefine
  {
    None,
    Base,
    Full
  };

  enum class InitialMesh
  {
    Interface,
    MaxRadius,
    Divisions
  };

  std::vector<unsigned int>
  get_primes(unsigned int start, unsigned int end);

  std::pair<unsigned int, unsigned int>
  decompose_to_prime_tuple(const unsigned int n_ref, const unsigned max_prime);

  template <int dim>
  void
  print_mesh_info(const Point<dim>                &bottom_left,
                  const Point<dim>                &top_right,
                  const std::vector<unsigned int> &subdivisions,
                  const unsigned int               n_refinements_global,
                  const unsigned int               n_refinements_delayed)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    pcout << "Create subdivided hyperrectangle [";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(top_right[i] - bottom_left[i]);

        if (i + 1 != dim)
          pcout << " x ";
      }
    pcout << "] = [";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(bottom_left[i]) << "..."
              << std::to_string(top_right[i]);

        if (i + 1 != dim)
          pcout << " x ";
      }
    pcout << "] " << std::endl;

    const unsigned int n_refinements =
      n_refinements_global + n_refinements_delayed;

    pcout << "with " << std::to_string(n_refinements) << " refinements (";
    pcout << "global = " << n_refinements_global << ", ";
    pcout << "delayed = " << n_refinements_delayed << ") and ";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(subdivisions[i]);

        if (i + 1 != dim)
          pcout << "x";
      }

    pcout << " subdivisions" << std::endl << std::endl;
  }

  namespace internal
  {
    /* The function reduces the initial number of subdivisions n_divs trying to
    minimize the following quality function: $q = n_{init} - n_{new} *
    2^{refines}$, where $n_{new}$ is a prime number. The algorithm does this
    operation for each of the dimensions independentaly. Aparently, this may
    lead to a situation, when the dimensions have different number of
    refinements. The minimum is then taken among the latter and the other
    $n_{new}$ are increased accordingly in order to comply with the chosen
    number of initial refinements. */
    unsigned int
    adjust_divisions_to_primes(const unsigned int         max_prime,
                               std::vector<unsigned int> &subdivisions);

    /* This function imposes peridoc boundary conditions in all dimensions of
     * the grid. */
    template <int dim, typename Triangulation>
    void
    impose_periodicity(Triangulation &tria)
    {
      // Need to work with triangulation here
      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(
            tria, 2 * d, 2 * d + 1, d, periodicity_vector);
        }

      tria.add_periodicity(periodicity_vector);
    }

    /* Adjust the initial refinements depending on the chosen refinement
    strategy. Two types of refinements are discussed: 1) base - this is the
    reference number of global refinements to reach the desirable coarse grid,
    that could have been further coarsened and decomposing to prime numbers. 2)
    interface - these is the reference number of global refinements to reach the
    desirable quality of the grid at diffuse interface. Depending on the
    refinement strategy, this function decides when and how these two
    refinements should be performed. 2) interface
    - `InitialRefine::Base`: the base refinements are performed as global here
    and the interface refinements are performed later by the AMR algo;
    - `InitialRefine::Full`: both base and interface refinements are performed
    as global here;
    - `InitialRefine::None`: no global refinements to be performed here and both
    base and interface refinements are performed later by the AMR algo. */
    template <typename Triangulation>
    std::pair<unsigned int, unsigned int>
    make_initial_refines(Triangulation      &tria,
                         const InitialRefine refine,
                         const unsigned int  n_refinements_base,
                         const unsigned int  n_refinements_interface)
    {
      unsigned int n_global  = 0;
      unsigned int n_delayed = 0;
      if (refine == InitialRefine::Base)
        {
          tria.refine_global(n_refinements_base);

          n_global  = n_refinements_base;
          n_delayed = n_refinements_interface;
        }
      else if (refine == InitialRefine::Full)
        {
          tria.refine_global(n_refinements_base + n_refinements_interface);

          n_global  = n_refinements_base + n_refinements_interface;
          n_delayed = 0;
        }
      else
        {
          n_global  = 0;
          n_delayed = n_refinements_base + n_refinements_interface;
        }

      return {n_global, n_delayed};
    }
  } // namespace internal

  /* This function creates a grid departing from the desirable cell size at
  diffuse interfaces. Based on this information, at first, the maximum numbers
  of subdivisions in each dimension are computed and then these are reduced by
  using the prime numbers decomposition. */
  template <typename Triangulation, int dim>
  unsigned int
  create_mesh_from_interface(
    Triangulation      &tria,
    const Point<dim>   &bottom_left,
    const Point<dim>   &top_right,
    const double        interface_width,
    const double        divisions_per_interface,
    const bool          periodic,
    const InitialRefine refine,
    const unsigned int  max_prime                          = 0,
    const double        max_level0_divisions_per_interface = 1.0,
    const unsigned int  divisions_per_element              = 1,
    const bool          print_stats                        = true)
  {
    // Domain size
    const auto domain_size = top_right - bottom_left;

    // Recompute divisions to elements
    const double elements_per_interface =
      static_cast<double>(divisions_per_interface) / divisions_per_element;
    const double max_level0_elements_per_interface =
      max_level0_divisions_per_interface / divisions_per_element;

    // Desirable smallest element size
    const double h_e = interface_width / elements_per_interface;

    // Number of refinements to get the desirable element size
    const unsigned int n_refinements_interface =
      static_cast<unsigned int>(std::ceil(
        std::log2(elements_per_interface / max_level0_elements_per_interface)));

    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int d = 0; d < dim; d++)
      {
        subdivisions[d] = static_cast<unsigned int>(std::ceil(
          domain_size[d] / h_e / std::pow(2, n_refinements_interface)));
      }

    // Further reduce the number of initial subdivisions
    unsigned int n_refinements_base =
      internal::adjust_divisions_to_primes(max_prime, subdivisions);

    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      internal::impose_periodicity<dim>(tria);

    const auto [n_global, n_delayed] = internal::make_initial_refines(
      tria, refine, n_refinements_base, n_refinements_interface);

    if (print_stats)
      print_mesh_info(
        bottom_left, top_right, subdivisions, n_global, n_delayed);

    return n_delayed;
  }

  /* This function creates a grid with the predefined number of sudivisions. */
  template <typename Triangulation, int dim>
  unsigned int
  create_mesh_from_divisions(Triangulation                   &tria,
                             const Point<dim>                &bottom_left,
                             const Point<dim>                &top_right,
                             const std::vector<unsigned int> &subdivisions,
                             const bool                       periodic,
                             const unsigned int               n_refinements,
                             const bool print_stats = true)
  {
    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      internal::impose_periodicity<dim>(tria);

    tria.refine_global(n_refinements);

    if (print_stats)
      print_mesh_info(bottom_left, top_right, subdivisions, n_refinements, 0);

    // Return 0 delayed lazy refinements for consistency of the interfaces
    return 0;
  }

  /* This function creates a grid by choosing the coarse one in such a way that
   * the cell size equals to the diameter of the largest particle of the
   * packing. This grid if then further coarsened by decomposing to prime
   * nubmers and refined afterwards to meet the requirements regarding the cell
   * sizes at the diffuse iterfaces. */
  template <typename Triangulation, int dim>
  unsigned int
  create_mesh_from_radius(Triangulation      &tria,
                          const Point<dim>   &bottom_left,
                          const Point<dim>   &top_right,
                          const double        interface_width,
                          const double        divisions_per_interface,
                          const double        r_ref,
                          const bool          periodic,
                          const InitialRefine refine,
                          const unsigned int  max_prime             = 0,
                          const unsigned int  divisions_per_element = 1,
                          const bool          print_stats           = true)
  {
    // Domain size
    const auto domain_size = top_right - bottom_left;

    // Recompute divisions to elements
    const double elements_per_interface =
      static_cast<double>(divisions_per_interface) / divisions_per_element;

    // Desirable smallest element size
    const double h_e = interface_width / elements_per_interface;

    // Reference size (diameter)
    const double h_ref = 2 * r_ref;

    // Number of refinements to get the desirable element size
    const unsigned int n_refinements_interface =
      static_cast<unsigned int>(std::ceil(std::log2(h_ref / h_e)));

    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int d = 0; d < dim; d++)
      subdivisions[d] =
        static_cast<unsigned int>(std::round(domain_size[d] / h_ref));

    // Try to reduce the number of initial subdivisions
    unsigned int n_refinements_base =
      internal::adjust_divisions_to_primes(max_prime, subdivisions);

    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      internal::impose_periodicity<dim>(tria);

    const auto [n_global, n_delayed] = internal::make_initial_refines(
      tria, refine, n_refinements_base, n_refinements_interface);

    if (print_stats)
      print_mesh_info(
        bottom_left, top_right, subdivisions, n_global, n_delayed);

    return n_delayed;
  }

  namespace internal
  {
    template <typename VectorType, typename Triangulation, int dim>
    void
    prepare_coarsening_and_refinement(
      const VectorType                     &solution_to_estimate,
      Triangulation                        &tria,
      DoFHandler<dim>                      &dof_handler,
      const Quadrature<dim - 1>            &quad,
      const double                          top_fraction_of_cells,
      const double                          bottom_fraction_of_cells,
      const unsigned int                    min_allowed_level,
      const unsigned int                    max_allowed_level,
      const typename VectorType::value_type val_min = 0.05,
      const typename VectorType::value_type val_max = 0.95)
    {
      // estimate errors
      Vector<float> estimated_error_per_cell(tria.n_active_cells());

      for (unsigned int b = 0; b < solution_to_estimate.n_blocks(); ++b)
        {
          Vector<float> estimated_error_per_cell_temp(tria.n_active_cells());

          KellyErrorEstimator<dim>::estimate(
            dof_handler,
            quad,
            std::map<types::boundary_id, const Function<dim> *>(),
            solution_to_estimate.block(b),
            estimated_error_per_cell_temp,
            {},
            nullptr,
            0,
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

          for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
            estimated_error_per_cell[i] += estimated_error_per_cell_temp[i] *
                                           estimated_error_per_cell_temp[i];
        }

      for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
        estimated_error_per_cell[i] = std::sqrt(estimated_error_per_cell[i]);

      // mark automatically cells for coarsening/refinement, ...
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        tria,
        estimated_error_per_cell,
        top_fraction_of_cells,
        bottom_fraction_of_cells);

      // make sure that cells close to the interfaces are refined, ...
      Vector<typename VectorType::value_type> values(
        dof_handler.get_fe().n_dofs_per_cell());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false || cell->refine_flag_set())
            continue;

          for (unsigned int b = 0; b < solution_to_estimate.n_blocks(); ++b)
            {
              cell->get_dof_values(solution_to_estimate.block(b), values);

              typename VectorType::value_type val_avg = 0;

              for (unsigned int i = 0; i < values.size(); ++i)
                {
                  val_avg += values[i];

                  if (val_min < values[i] && values[i] < val_max)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();

                      break;
                    }
                }

              if (!cell->refine_flag_set())
                {
                  // In case if a cell has values, e.g., close to 0 or 1
                  val_avg /= values.size();
                  if (val_min < val_avg && val_avg < val_max)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();
                    }
                }

              if (cell->refine_flag_set())
                break;
            }
        }

      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto cell_level = static_cast<unsigned int>(cell->level());

          if (cell->refine_flag_set() && cell_level == max_allowed_level)
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() && cell_level == min_allowed_level)
            cell->clear_coarsen_flag();

          // Coarsen cell if it is overrefined
          if (cell_level > max_allowed_level)
            {
              if (cell->refine_flag_set())
                cell->clear_refine_flag();
              cell->set_coarsen_flag();
            }
        }

      tria.prepare_coarsening_and_refinement();
    }
  } // namespace internal

  template <typename VectorType, typename Triangulation, int dim>
  void
  coarsen_and_refine_mesh(const VectorType          &solution_to_estimate,
                          Triangulation             &tria,
                          DoFHandler<dim>           &dof_handler,
                          const Quadrature<dim - 1> &quad,
                          const double               top_fraction_of_cells,
                          const double               bottom_fraction_of_cells,
                          const unsigned int         min_allowed_level,
                          const unsigned int         max_allowed_level,
                          const typename VectorType::value_type val_min = 0.05,
                          const typename VectorType::value_type val_max = 0.95)
  {
    // Mark cells and prepare refinement
    internal::prepare_coarsening_and_refinement(solution_to_estimate,
                                                tria,
                                                dof_handler,
                                                quad,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells,
                                                min_allowed_level,
                                                max_allowed_level,
                                                val_min,
                                                val_max);

    // Execute refinement
    tria.execute_coarsening_and_refinement();
  }

  template <typename VectorType, typename Triangulation, int dim>
  void
  coarsen_and_refine_mesh(
    VectorType                                         &solution,
    Triangulation                                      &tria,
    DoFHandler<dim>                                    &dof_handler,
    AffineConstraints<typename VectorType::value_type> &constraints,
    const Quadrature<dim - 1>                          &quad,
    const double                                        top_fraction_of_cells,
    const double                          bottom_fraction_of_cells,
    const unsigned int                    min_allowed_level,
    const unsigned int                    max_allowed_level,
    std::function<void(VectorType &)>     reinitializer,
    const typename VectorType::value_type val_min              = 0.05,
    const typename VectorType::value_type val_max              = 0.95,
    const unsigned int                    block_estimate_start = 0,
    const unsigned int block_estimate_end = numbers::invalid_unsigned_int)
  {
    // Build partitioner
    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_mpi_communicator());

    // Copy solution so that it has the right ghosting
    VectorType solution_copy(solution.n_blocks());
    for (unsigned int b = 0; b < solution_copy.n_blocks(); ++b)
      {
        solution_copy.block(b).reinit(partitioner);
        solution_copy.block(b).copy_locally_owned_data_from(solution.block(b));
      }

    for (unsigned int b = 0; b < solution_copy.n_blocks(); ++b)
      constraints.distribute(solution_copy.block(b));

    solution_copy.update_ghost_values();

    const auto solution_copy_estimate =
      solution_copy.create_view(block_estimate_start,
                                block_estimate_end !=
                                    numbers::invalid_unsigned_int ?
                                  block_estimate_end :
                                  solution_copy.n_blocks());

    // Mark cells and prepare refinement
    internal::prepare_coarsening_and_refinement(*solution_copy_estimate,
                                                tria,
                                                dof_handler,
                                                quad,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells,
                                                min_allowed_level,
                                                max_allowed_level,
                                                val_min,
                                                val_max);

    // Raw pointers
    std::vector<typename VectorType::BlockType *> solution_ptr(
      solution.n_blocks());
    std::vector<const typename VectorType::BlockType *> solution_copy_ptr(
      solution_copy.n_blocks());
    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
      {
        solution_ptr[b]      = &solution.block(b);
        solution_copy_ptr[b] = &solution_copy.block(b);
      }

    // Prepare solution transfer
    parallel::distributed::SolutionTransfer<dim, typename VectorType::BlockType>
      solution_trans(dof_handler);

    solution_trans.prepare_for_coarsening_and_refinement(solution_copy_ptr);

    // Execute refinement
    tria.execute_coarsening_and_refinement();

    // Reinitialize vector to be transfered
    reinitializer(solution);

    // Transfer solution
    solution_trans.interpolate(solution_ptr);

    // note: apply constraints since the Newton solver expects this
    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
      constraints.distribute(solution.block(b));
  }
} // namespace hpsint