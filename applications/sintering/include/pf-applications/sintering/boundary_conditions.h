#pragma once

#include <deal.II/base/point.h>

#include <deal.II/grid/grid_tools.h>

#include <pf-applications/sintering/sintering_data.h>

#include <pf-applications/grain_tracker/tracker.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim,
            typename Number,
            typename VectorType,
            typename VectorizedArrayType>
  void
  clamp_section(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const VectorType &                                  concentration,
    const Point<dim> &                                  origin,
    const unsigned int                                  direction = 0)
  {
    concentration.update_ghost_values();

    std::vector<types::global_dof_index> local_face_dof_indices(
      matrix_free.get_dofs_per_face());
    std::set<types::global_dof_index> indices_to_add;

    double       c_max_on_face    = 0.;
    unsigned int id_c_max_on_face = numbers::invalid_unsigned_int;

    // Apply constraints for displacement along the direction axis
    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (const auto &cell :
         matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto &face : cell->face_iterators())
          if (std::abs(face->center()(direction) - origin[direction]) < 1e-9)
            {
              face->get_dof_indices(local_face_dof_indices);

              for (const auto i : local_face_dof_indices)
                {
                  const auto local_index = partitioner->global_to_local(i);
                  indices_to_add.insert(local_index);

                  if (concentration.local_element(local_index) > c_max_on_face)
                    {
                      c_max_on_face = concentration.local_element(local_index);
                      id_c_max_on_face = local_index;
                    }
                }
            }

    const auto comm = matrix_free.get_dof_handler().get_communicator();

    const double global_c_max_on_face =
      Utilities::MPI::max(c_max_on_face, comm);

    unsigned int rank_having_c_max =
      std::abs(global_c_max_on_face - c_max_on_face) < 1e-16 ?
        Utilities::MPI::this_mpi_process(comm) :
        numbers::invalid_unsigned_int;
    rank_having_c_max = Utilities::MPI::min(rank_having_c_max, comm);

    // Remove previous costraionts
    for (unsigned int d = 0; d < dim; ++d)
      displ_constraints_indices[d].clear();

    // Add cross-section constraints
    std::copy(indices_to_add.begin(),
              indices_to_add.end(),
              std::back_inserter(displ_constraints_indices[direction]));

    // Add pointwise constraints
    bool add_pointwise =
      rank_having_c_max == Utilities::MPI::this_mpi_process(comm);
    if (add_pointwise)
      for (unsigned int d = 0; d < dim; ++d)
        if (d != direction)
          displ_constraints_indices[d].push_back(id_c_max_on_face);

    concentration.zero_out_ghost_values();
  }

  template <int dim,
            typename Number,
            typename VectorType,
            typename VectorizedArrayType>
  void
  clamp_central_section(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const VectorType &                                  concentration,
    const unsigned int                                  direction = 0)
  {
    // Add central constraints
    const auto bb_tria = GridTools::compute_bounding_box(
      matrix_free.get_dof_handler().get_triangulation());

    auto center = bb_tria.get_boundary_points().first +
                  bb_tria.get_boundary_points().second;
    center /= 2.;

    clamp_section<dim>(
      displ_constraints_indices, matrix_free, concentration, center, direction);
  }

  template <int dim,
            typename Number,
            typename BlockVectorType,
            typename VectorizedArrayType>
  void
  clamp_section_within_particle(
    std::array<std::vector<unsigned int>, dim> &displ_constraints_indices,
    const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
    const SinteringOperatorData<dim, VectorizedArrayType> &data,
    const GrainTracker::Tracker<dim, Number> &             grain_tracker,
    const BlockVectorType &                                solution,
    const Point<dim> &                                     origin_in,
    const unsigned int                                     direction = 0,
    const Number order_parameter_threshold                           = 0.1)
  {
    for (unsigned int b = 2; b < data.n_components(); ++b)
      solution.block(b).update_ghost_values();

    const auto &partitioner = matrix_free.get_vector_partitioner();

    const auto comm = matrix_free.get_dof_handler().get_communicator();

    // Find closest vertex, the corresponding vertex index and containing cell
    const auto containing_cell = GridTools::find_active_cell_around_point(
      matrix_free.get_dof_handler().get_triangulation(), origin_in);

    types::global_dof_index global_vertex_index = numbers::invalid_unsigned_int;
    Point<dim>              origin;

    if (containing_cell->is_locally_owned())
      {
        Number dist_min = std::numeric_limits<Number>::max();

        for (unsigned int v = 0; v < containing_cell->n_vertices(); ++v)
          {
            const auto dist = origin_in.distance(containing_cell->vertex(v));
            if (dist < dist_min)
              {
                global_vertex_index = containing_cell->vertex_index(v);
                origin              = containing_cell->vertex(v);
                dist_min            = dist;
              }
          }
      }

    // What rank actualy owns this vertex
    unsigned int rank_having_vertex =
      global_vertex_index != numbers::invalid_unsigned_int ?
        Utilities::MPI::this_mpi_process(comm) :
        numbers::invalid_unsigned_int;
    rank_having_vertex = Utilities::MPI::min(rank_having_vertex, comm);

    // Broadcast origin point to all ranks
    origin = Utilities::MPI::broadcast(comm, origin, rank_having_vertex);

    // The owner of the origin finds the corresponding order parameter and
    // particle ids
    unsigned int primary_order_parameter_id = numbers::invalid_unsigned_int;
    unsigned int primary_particle_id        = numbers::invalid_unsigned_int;

    if (global_vertex_index != numbers::invalid_unsigned_int)
      {
        const auto cell_index = containing_cell->global_active_cell_index();
        for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
          {
            const auto particle_id_for_op =
              grain_tracker.get_particle_index(ig, cell_index);

            if (particle_id_for_op != numbers::invalid_unsigned_int)
              {
                if (primary_order_parameter_id == numbers::invalid_unsigned_int)
                  {
                    primary_order_parameter_id = ig;
                    primary_particle_id        = particle_id_for_op;
                  }
                else
                  {
                    AssertThrow(
                      false,
                      ExcMessage(
                        "Multiple particles located at the origin point, "
                        "the clamping constraints can be imposed only at "
                        "points which are not shared"));
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
    std::set<types::global_dof_index> indices_to_add;

    // Apply constraints for displacement along the direction axis
    const auto &concentration = solution.block(primary_order_parameter_id + 2);

    double       c_max_on_face    = 0.;
    unsigned int id_c_max_on_face = numbers::invalid_unsigned_int;

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
              if (std::abs(face->center()(direction) - origin[direction]) <
                  1e-9)
                {
                  face->get_dof_indices(local_face_dof_indices);

                  for (const auto i : local_face_dof_indices)
                    {
                      const auto local_index = partitioner->global_to_local(i);
                      const auto concentration_local =
                        concentration.local_element(local_index);

                      // Restrain only points inside a particle
                      if (concentration_local > order_parameter_threshold)
                        {
                          indices_to_add.insert(local_index);

                          if (concentration_local > c_max_on_face)
                            {
                              c_max_on_face    = concentration_local;
                              id_c_max_on_face = local_index;
                            }
                        }
                    }
                }
        }

    const double global_c_max_on_face =
      Utilities::MPI::max(c_max_on_face, comm);

    unsigned int rank_having_c_max =
      std::abs(global_c_max_on_face - c_max_on_face) < 1e-16 ?
        Utilities::MPI::this_mpi_process(comm) :
        numbers::invalid_unsigned_int;
    rank_having_c_max = Utilities::MPI::min(rank_having_c_max, comm);

    // Remove previous costraionts
    for (unsigned int d = 0; d < dim; ++d)
      displ_constraints_indices[d].clear();

    // Add cross-section constraints
    std::copy(indices_to_add.begin(),
              indices_to_add.end(),
              std::back_inserter(displ_constraints_indices[direction]));

    // Add pointwise constraints
    bool add_pointwise =
      rank_having_c_max == Utilities::MPI::this_mpi_process(comm);
    if (add_pointwise)
      for (unsigned int d = 0; d < dim; ++d)
        if (d != direction)
          displ_constraints_indices[d].push_back(id_c_max_on_face);

    for (unsigned int b = 2; b < data.n_components(); ++b)
      solution.block(b).zero_out_ghost_values();
  }

  template <int dim, typename Number>
  Point<dim>
  find_center_origin(const Triangulation<dim> &                triangulation,
                     const GrainTracker::Tracker<dim, Number> &grain_tracker,
                     const bool prefer_growing = false)
  {
    // Add central constraints
    const auto bb_tria = GridTools::compute_bounding_box(triangulation);
    const auto center  = bb_tria.center();

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