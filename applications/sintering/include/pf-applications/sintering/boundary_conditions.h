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

#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/grid_tools.h>

#include <pf-applications/base/geometry.h>

#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/tracker.h>

namespace Sintering
{
  using namespace dealii;

  namespace internal
  {
    template <int dim, typename Number, typename VectorizedArrayType>
    std::tuple<typename Triangulation<dim>::active_cell_iterator,
               Point<dim>,
               unsigned int,
               unsigned int>
    find_containing_cell_and_dof_index(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const Mapping<dim> &                                mapping,
      const Point<dim> &                                  origin_in)
    {
      const auto   comm = matrix_free.get_dof_handler().get_communicator();
      const double tol  = 1e-10;

      // Find closest vertex, the corresponding vertex index and containing cell
      const auto first_pair = GridTools::find_active_cell_around_point(
        mapping,
        matrix_free.get_dof_handler().get_triangulation(),
        origin_in,
        {},
        tol);

      auto containing_cell = first_pair.first;

      // Check if any rank found a locally_owned_cell
      const bool is_locally_owned =
        containing_cell !=
          matrix_free.get_dof_handler().get_triangulation().end() &&
        containing_cell->is_locally_owned();

      // What rank actualy owns this vertex
      unsigned int rank_having_locally_owned =
        is_locally_owned ? Utilities::MPI::this_mpi_process(comm) :
                           numbers::invalid_unsigned_int;
      rank_having_locally_owned =
        Utilities::MPI::min(rank_having_locally_owned, comm);

      // If all ranks have identified ghost cells only
      if (rank_having_locally_owned == numbers::invalid_unsigned_int)
        {
          if (containing_cell !=
              matrix_free.get_dof_handler().get_triangulation().end())
            {
              auto all_cells = GridTools::find_all_active_cells_around_point(
                mapping,
                matrix_free.get_dof_handler().get_triangulation(),
                origin_in,
                tol,
                first_pair);

              // Iterate through all the found cells until a locally owned one
              // is found
              auto it_locally_owned_pair = all_cells.cbegin();
              for (; it_locally_owned_pair != all_cells.cend() &&
                     !it_locally_owned_pair->first->is_locally_owned();
                   ++it_locally_owned_pair)
                ;

              // If a locally owned cell was found
              if (it_locally_owned_pair != all_cells.cend())
                {
                  containing_cell = it_locally_owned_pair->first;
                  rank_having_locally_owned =
                    Utilities::MPI::this_mpi_process(comm);
                }
            }

          rank_having_locally_owned =
            Utilities::MPI::min(rank_having_locally_owned, comm);
        }

      AssertThrow(
        rank_having_locally_owned != numbers::invalid_unsigned_int,
        ExcMessage(
          "Failed to identify a locally owned cell for the origin point"));

      types::global_dof_index global_scalar_dof_index =
        numbers::invalid_unsigned_int;
      Point<dim> origin;

      // If the current rank is assigned to do the job
      if (rank_having_locally_owned == Utilities::MPI::this_mpi_process(comm))
        {
          auto dist_min = std::numeric_limits<double>::max();

          for (unsigned int v = 0; v < containing_cell->n_vertices(); ++v)
            {
              const auto dist = origin_in.distance(containing_cell->vertex(v));
              if (dist < dist_min)
                {
                  origin   = containing_cell->vertex(v);
                  dist_min = dist;

                  const auto dof_cell = typename DoFHandler<dim>::cell_iterator(
                    *containing_cell, &matrix_free.get_dof_handler());
                  global_scalar_dof_index = dof_cell->vertex_dof_index(v, 0);
                }
            }
        }

      // Broadcast the origin point to all ranks
      origin =
        Utilities::MPI::broadcast(comm, origin, rank_having_locally_owned);

      return std::make_tuple(containing_cell,
                             origin,
                             global_scalar_dof_index,
                             rank_having_locally_owned);
    }
  } // namespace internal

  template <int dim, typename Number, typename VectorizedArrayType>
  void
  clamp_section(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const Mapping<dim> &                                mapping,
    const Point<dim> &                                  origin_in,
    const std::array<bool, dim> &                       directions_mask)
  {
    std::vector<unsigned int> directions;
    for (unsigned int d = 0; d < dim; ++d)
      if (directions_mask[d])
        directions.push_back(d);

    const auto comm = matrix_free.get_dof_handler().get_communicator();

    const auto [containing_cell,
                origin_global,
                global_scalar_dof_index,
                rank_having_vertex] =
      internal::find_containing_cell_and_dof_index(matrix_free,
                                                   mapping,
                                                   origin_in);

    std::vector<types::global_dof_index> local_face_dof_indices(
      matrix_free.get_dofs_per_face());
    std::array<std::set<types::global_dof_index>, dim> indices_to_add;

    // Apply constraints for displacement along the direction axis
    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (const auto &cell :
         matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &face : cell->face_iterators())
          {
            face->get_dof_indices(local_face_dof_indices);

            // Iterate over all directions
            for (const auto &d : directions)
              if (std::abs(face->center()(d) - origin_global[d]) < 1e-9)
                {
                  for (const auto i : local_face_dof_indices)
                    {
                      const auto local_index = partitioner->global_to_local(i);
                      indices_to_add[d].insert(local_index);
                    }
                }
          }

    // Remove previous costraionts
    for (unsigned int d = 0; d < dim; ++d)
      {
        displ_constraints_indices[d].clear();

        // Add cross-section constraints
        std::copy(indices_to_add[d].begin(),
                  indices_to_add[d].end(),
                  std::back_inserter(displ_constraints_indices[d]));
      }

    // Add pointwise constraints
    bool add_pointwise =
      rank_having_vertex == Utilities::MPI::this_mpi_process(comm) &&
      directions.size() < (dim - 1);
    if (add_pointwise)
      {
        const auto local_dof_index =
          partitioner->global_to_local(global_scalar_dof_index);

        for (unsigned int d = 0; d < dim; ++d)
          if (!directions_mask[d])
            displ_constraints_indices[d].push_back(local_dof_index);
      }
  }

  template <int dim, typename Number, typename VectorizedArrayType>
  void
  clamp_central_section(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const Mapping<dim> &                                mapping,
    const std::array<bool, dim> &                       directions_mask)
  {
    // Add central constraints
    const auto bb_tria = GridTools::compute_bounding_box(
      matrix_free.get_dof_handler().get_triangulation());

    auto center = bb_tria.get_boundary_points().first +
                  bb_tria.get_boundary_points().second;
    center /= 2.;

    clamp_section<dim>(
      displ_constraints_indices, matrix_free, mapping, center, directions_mask);
  }

  template <int dim,
            typename Number,
            typename BlockVectorType,
            typename VectorizedArrayType>
  void
  clamp_section_within_particle(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
    const Mapping<dim> &                                   mapping,
    const SinteringOperatorData<dim, VectorizedArrayType> &data,
    const GrainTracker::Tracker<dim, Number> &             grain_tracker,
    const BlockVectorType &                                solution,
    const Point<dim> &                                     origin_in,
    const std::array<bool, dim> &                          directions_mask,
    const Number order_parameter_threshold = 0.1)
  {
    for (unsigned int b = 2; b < data.n_components(); ++b)
      solution.block(b).update_ghost_values();

    std::vector<unsigned int> directions;
    for (unsigned int d = 0; d < dim; ++d)
      if (directions_mask[d])
        directions.push_back(d);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    const auto comm = matrix_free.get_dof_handler().get_communicator();

    const auto [containing_cell,
                origin_global,
                global_scalar_dof_index,
                rank_having_vertex] =
      internal::find_containing_cell_and_dof_index(matrix_free,
                                                   mapping,
                                                   origin_in);

    // The owner of the origin finds the corresponding order parameter and
    // particle ids. It may turn out that a particular particle, though being at
    // the center of the domain, but is shrinking fast at the moment. Using it
    // as the reference one is not a reliable option. In such cases multiple
    // particle are detected at the origin point and the largest one is picked.
    unsigned int primary_order_parameter_id = numbers::invalid_unsigned_int;
    unsigned int primary_particle_id        = numbers::invalid_unsigned_int;
    double       primary_grain_measure      = 0;

    if (global_scalar_dof_index != numbers::invalid_unsigned_int)
      {
        const auto cell_index = containing_cell->global_active_cell_index();
        for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
          {
            const auto particle_id_for_op =
              grain_tracker.get_particle_index(ig, cell_index);

            if (particle_id_for_op != numbers::invalid_unsigned_int)
              {
                const auto grain_id =
                  grain_tracker.get_grain_and_segment(ig, particle_id_for_op)
                    .first;

                const auto grain_measure =
                  grain_tracker.get_grains().at(grain_id).get_measure();

                if (grain_measure > primary_grain_measure)
                  {
                    primary_order_parameter_id = ig;
                    primary_particle_id        = particle_id_for_op;
                    primary_grain_measure      = grain_measure;
                  }
              }
          }

        AssertThrow(primary_order_parameter_id !=
                        numbers::invalid_unsigned_int &&
                      primary_particle_id != numbers::invalid_unsigned_int,
                    ExcMessage("No particle detected at the origin point"));
      }

    // Broadcast order parameter and particle ids to all ranks
    primary_order_parameter_id =
      Utilities::MPI::broadcast(comm,
                                primary_order_parameter_id,
                                rank_having_vertex);
    primary_particle_id =
      Utilities::MPI::broadcast(comm, primary_particle_id, rank_having_vertex);

    // Prepare structures for collecting indices
    std::vector<types::global_dof_index> local_face_dof_indices(
      matrix_free.get_dofs_per_face());
    std::array<std::set<types::global_dof_index>, dim> indices_to_add;

    // Apply constraints for displacement along the direction axis
    const auto &concentration = solution.block(primary_order_parameter_id + 2);

    for (const auto &cell :
         matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto cell_index = cell->global_active_cell_index();

          const unsigned int particle_id =
            grain_tracker.get_particle_index(primary_order_parameter_id,
                                             cell_index);

          if (particle_id == primary_particle_id)
            for (const auto &face : cell->face_iterators())
              {
                face->get_dof_indices(local_face_dof_indices);

                // Iterate over all directions
                for (const auto &d : directions)
                  if (std::abs(face->center()(d) - origin_global[d]) < 1e-9)
                    {
                      for (const auto i : local_face_dof_indices)
                        {
                          const auto local_index =
                            partitioner->global_to_local(i);
                          const auto concentration_local =
                            concentration.local_element(local_index);

                          // Restrain only points inside a particle
                          if (concentration_local > order_parameter_threshold)
                            {
                              indices_to_add[d].insert(local_index);
                            }
                        }
                    }
              }
        }

    // Remove previous costraionts
    for (unsigned int d = 0; d < dim; ++d)
      {
        displ_constraints_indices[d].clear();

        // Add cross-section constraints
        std::copy(indices_to_add[d].begin(),
                  indices_to_add[d].end(),
                  std::back_inserter(displ_constraints_indices[d]));
      }

    // Add pointwise constraints
    bool add_pointwise =
      rank_having_vertex == Utilities::MPI::this_mpi_process(comm) &&
      directions.size() < dim;
    if (add_pointwise)
      {
        const auto local_dof_index =
          partitioner->global_to_local(global_scalar_dof_index);

        for (unsigned int d = 0; d < dim; ++d)
          if (!directions_mask[d])
            displ_constraints_indices[d].push_back(local_dof_index);
      }

    for (unsigned int b = 2; b < data.n_components(); ++b)
      solution.block(b).zero_out_ghost_values();
  }

  template <int dim, typename Number>
  Point<dim>
  find_center_origin(const Triangulation<dim> &                triangulation,
                     const GrainTracker::Tracker<dim, Number> &grain_tracker,
                     const bool prefer_growing = false,
                     const bool use_barycenter = false)
  {
    // Add central constraints
    const auto bb_tria = GridTools::compute_bounding_box(triangulation);

    const auto center =
      use_barycenter ?
        GrainTracker::calc_cloud_barycenter(grain_tracker.get_grains()) :
        bb_tria.center();

    Point<dim> origin;

    Number dist_min = std::numeric_limits<Number>::max();

    typename GrainTracker::Grain<dim>::Dynamics dynamics_max =
      GrainTracker::Grain<dim>::None;

    for (const auto &[grain_id, grain] : grain_tracker.get_grains())
      for (const auto &segment : grain.get_segments())
        {
          const Number dist = segment.get_center().distance(center);

          const bool pick =
            (!prefer_growing && dist < dist_min) ||
            (prefer_growing &&
             ((dist < dist_min && grain.get_dynamics() >= dynamics_max) ||
              dist_min == std::numeric_limits<Number>::max()));

          if (pick)
            {
              dist_min     = dist;
              origin       = segment.get_center();
              dynamics_max = grain.get_dynamics();
            }
        }

    return origin;
  }

  template <int dim, typename Number, typename VectorizedArrayType>
  void
  clamp_domain(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
  {
    const auto &partitioner = matrix_free.get_vector_partitioner();

    // Remove previous costraionts
    for (unsigned int d = 0; d < dim; ++d)
      displ_constraints_indices[d].clear();

    std::vector<types::global_dof_index> local_face_dof_indices(
      matrix_free.get_dofs_per_face());

    for (const auto &cell :
         matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            for (unsigned int d = 0; d < dim; ++d)
              // Default colorization is implied
              if (face->boundary_id() == 2 * d)
                {
                  face->get_dof_indices(local_face_dof_indices);

                  for (const auto i : local_face_dof_indices)
                    {
                      const auto local_index = partitioner->global_to_local(i);
                      displ_constraints_indices[d].push_back(local_index);
                    }
                }
  }


} // namespace Sintering