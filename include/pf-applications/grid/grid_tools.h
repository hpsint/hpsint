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

#include <deal.II/base/partitioner.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include "bounding_box_filter.h"

using namespace dealii;

namespace dealii
{
  template <int dim, typename VectorType>
  bool
  coarsen_triangulation(parallel::distributed::Triangulation<dim> &tria_copy,
                        const DoFHandler<dim> &background_dof_handler,
                        DoFHandler<dim> &      background_dof_handler_coarsened,
                        const VectorType &     vector,
                        VectorType &           vector_coarsened,
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
  update_selected_ghosts(VectorType &                  vector,
                         const VectorOperation::values operation,
                         Utilities::MPI::Partitioner & partitioner,
                         std::vector<Number> &         ghosts_values,
                         const IndexSet &              ghost_indices,
                         const IndexSet &              larger_ghost_index_set)
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
    const DoFHandler<dim> &                       background_dof_handler,
    VectorType &                                  vector,
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
      background_dof_handler.get_communicator());

    auto partitioner_reduced = std::make_shared<Utilities::MPI::Partitioner>(
      background_dof_handler.locally_owned_dofs(),
      background_dof_handler.get_communicator());

    using Number = typename VectorType::value_type;

    // Make local constraints
    AffineConstraints<Number> constraints;
    const auto                relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(background_dof_handler);

    constraints.clear();
    constraints.reinit(relevant_dofs);
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
} // namespace dealii