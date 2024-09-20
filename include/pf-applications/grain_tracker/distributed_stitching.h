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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/dofs/dof_handler.h>

#include <pf-applications/base/scoped_name.h>
#include <pf-applications/base/timer.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <pf-applications/grain_tracker/motion.h>

namespace GrainTracker
{
  using namespace dealii;

  template <int dim, typename VectorSolution, typename VectorIds>
  unsigned int
  run_flooding(const typename DoFHandler<dim>::cell_iterator &cell,
               const VectorSolution &                         solution,
               VectorIds &                                    particle_ids,
               const unsigned int                             id,
               double &                                       max_value,
               const double threshold_lower     = 0,
               const double invalid_particle_id = -1.0)
  {
    if (cell->has_children())
      {
        unsigned int counter = 0;

        for (const auto &child : cell->child_iterators())
          counter += run_flooding<dim>(child,
                                       solution,
                                       particle_ids,
                                       id,
                                       max_value,
                                       threshold_lower,
                                       invalid_particle_id);

        return counter;
      }

    if (cell->is_locally_owned() == false)
      return 0;

    const auto particle_id = particle_ids[cell->global_active_cell_index()];

    if (particle_id != invalid_particle_id)
      return 0; // cell has been visited

    Vector<double> values(cell->get_fe().n_dofs_per_cell());

    cell->get_dof_values(solution, values);

    const auto cell_max_value = *std::max_element(values.begin(), values.end());
    const bool has_particle   = cell_max_value > threshold_lower;

    if (!has_particle)
      return 0; // cell has no particle

    particle_ids[cell->global_active_cell_index()] = id;

    max_value = std::max(max_value, cell_max_value);

    unsigned int counter = 1;

    for (const auto face : cell->face_indices())
      if (cell->at_boundary(face) == false)
        counter += run_flooding<dim>(cell->neighbor(face),
                                     solution,
                                     particle_ids,
                                     id,
                                     max_value,
                                     threshold_lower,
                                     invalid_particle_id);

    return counter;
  }

  std::vector<unsigned int>
  connected_components(
    const unsigned int                                         N,
    const std::vector<std::tuple<unsigned int, unsigned int>> &edges)
  {
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>
                                                          Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

    Graph G(N);
    for (const auto &edge : edges)
      boost::add_edge(std::get<0>(edge), std::get<1>(edge), G);

    std::vector<unsigned int> c(boost::num_vertices(G));
    int                       num = boost::connected_components(
      G,
      make_iterator_property_map(c.begin(),
                                 boost::get(boost::vertex_index, G),
                                 c[0]));

    (void)num;

    return c;
  }

  std::vector<unsigned int>
  perform_distributed_stitching_via_graph(
    const MPI_Comm comm,
    const std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
      &            edges_in,
    MyTimerOutput *timer = nullptr)
  {
    ScopedName sc("perform_distributed_stitching");
    MyScope    scope(sc, timer);

    const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    const auto edges_temp = Utilities::MPI::gather(comm, edges_in, 0);

    std::vector<unsigned int> colors;
    std::vector<int>          sizes;
    std::vector<int>          offsets;

    if (my_rank == 0)
      {
        unsigned int N = 0;

        std::vector<std::tuple<unsigned int, unsigned int>> edges;

        sizes.assign(n_procs, 0);
        offsets.assign(n_procs, 0);

        for (unsigned int i = 0; i < edges_temp.size(); ++i)
          {
            for (unsigned int j = 0; j < edges_temp[i].size(); ++j)
              for (unsigned int k = 0; k < edges_temp[i][j].size(); ++k)
                edges.emplace_back(N + j, std::get<1>(edges_temp[i][j][k]));
            N += edges_temp[i].size();
            sizes[i] = edges_temp[i].size();
          }

        for (unsigned int i = 1; i < n_procs; ++i)
          offsets[i] = offsets[i - 1] + sizes[i - 1];

        colors = connected_components(N, edges);
      }

    std::vector<unsigned int> my_colors(edges_in.size());

    MPI_Scatterv(colors.data(),
                 sizes.data(),
                 offsets.data(),
                 MPI_UNSIGNED,
                 my_colors.data(),
                 my_colors.size(),
                 MPI_INT,
                 0,
                 comm);

    return my_colors;
  }

  std::vector<unsigned int>
  perform_distributed_stitching(
    const MPI_Comm                                                   comm,
    std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input,
    MyTimerOutput *timer = nullptr)
  {
    ScopedName sc("perform_distributed_stitching");
    MyScope    scope(sc, timer);

    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);
    const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

    // step 1) determine - via fixed-point iteration - the clique of
    // each particle
    const unsigned int local_size = input.size();
    unsigned int       offset     = 0;

    MPI_Exscan(&local_size, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

    using T = std::vector<
      std::tuple<unsigned int,
                 std::vector<std::tuple<unsigned int, unsigned int>>>>;

    unsigned int iter = 0;

    while (true)
      {
        ScopedName sc("fp_iter_" + std::to_string(iter));
        MyScope    scope(sc, timer);

        std::map<unsigned int, T> data_to_send;

        for (unsigned int i = 0; i < input.size(); ++i)
          {
            const auto input_i = input[i];
            for (unsigned int j = 0; j < input_i.size(); ++j)
              {
                const unsigned int other_rank = std::get<0>(input_i[j]);

                if (other_rank == my_rank)
                  continue;

                std::vector<std::tuple<unsigned int, unsigned int>> temp;

                temp.emplace_back(my_rank, i + offset);

                for (unsigned int k = 0; k < input_i.size(); ++k)
                  if (k != j)
                    temp.push_back(input_i[k]);

                std::sort(temp.begin(), temp.end());

                data_to_send[other_rank].emplace_back(std::get<1>(input_i[j]),
                                                      temp);
              }
          }

        bool finished = true;

        Utilities::MPI::ConsensusAlgorithms::selector<T>(
          [&]() {
            std::vector<unsigned int> targets;
            for (const auto &i : data_to_send)
              targets.emplace_back(i.first);
            return targets;
          }(),
          [&](const unsigned int other_rank) {
            return data_to_send[other_rank];
          },
          [&](const unsigned int, const auto &data) {
            for (const auto &data_i : data)
              {
                const unsigned int index   = std::get<0>(data_i) - offset;
                const auto &       values  = std::get<1>(data_i);
                auto &             input_i = input[index];

                const unsigned int old_size = input_i.size();

                input_i.insert(input_i.end(), values.begin(), values.end());
                std::sort(input_i.begin(), input_i.end());
                input_i.erase(std::unique(input_i.begin(), input_i.end()),
                              input_i.end());

                const unsigned int new_size = input_i.size();

                finished &= (old_size == new_size);
              }
          },
          comm);

        if (Utilities::MPI::sum(static_cast<unsigned int>(finished), comm) ==
            n_ranks) // run as long as no clique has changed
          break;

        ++iter;
      }

    // step 2) give each clique a unique id
    std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
      input_valid;

    for (unsigned int i = 0; i < input.size(); ++i)
      {
        auto input_i = input[i];

        if (input_i.size() == 0)
          {
            std::vector<std::tuple<unsigned int, unsigned int>> temp;
            temp.emplace_back(my_rank, i + offset);
            input_valid.push_back(temp);
          }
        else
          {
            if ((my_rank <= std::get<0>(input_i[0])) &&
                ((i + offset) < std::get<1>(input_i[0])))
              {
                input_i.insert(
                  input_i.begin(),
                  std::tuple<unsigned int, unsigned int>{my_rank, i + offset});
                input_valid.push_back(input_i);
              }
          }
      }

    // step 3) notify each particle of the id of its clique
    const unsigned int local_size_p = input_valid.size();
    unsigned int       offset_p     = 0;

    MPI_Exscan(&local_size_p, &offset_p, 1, MPI_UNSIGNED, MPI_SUM, comm);

    using U = std::vector<std::tuple<unsigned int, unsigned int>>;
    std::map<unsigned int, U> data_to_send_;

    for (unsigned int i = 0; i < input_valid.size(); ++i)
      {
        for (const auto &j : input_valid[i])
          data_to_send_[std::get<0>(j)].emplace_back(std::get<1>(j),
                                                     i + offset_p);
      }

    std::vector<unsigned int> result(input.size(),
                                     numbers::invalid_unsigned_int);

    {
      ScopedName sc("notify");
      MyScope    scope(sc, timer);
      Utilities::MPI::ConsensusAlgorithms::selector<U>(
        [&]() {
          std::vector<unsigned int> targets;
          for (const auto &i : data_to_send_)
            targets.emplace_back(i.first);
          return targets;
        }(),
        [&](const unsigned int other_rank) {
          return data_to_send_[other_rank];
        },
        [&](const unsigned int, const auto &data) {
          for (const auto &i : data)
            {
              AssertDimension(result[std::get<0>(i) - offset],
                              numbers::invalid_unsigned_int);
              result[std::get<0>(i) - offset] = std::get<1>(i);
            }
        },
        comm);
    }

    MPI_Barrier(comm);

    return result;
  }

  template <int dim, typename VectorIds>
  auto
  build_local_connectivity(const DoFHandler<dim> &dof_handler,
                           const VectorIds &      particle_ids,
                           const double           local_grains_num,
                           const double           local_offset,
                           const double           invalid_particle_id = -1.0)
  {
    std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
      local_connectivity(local_grains_num);

    for (const auto &ghost_cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (ghost_cell->is_ghost())
        {
          const auto particle_id =
            particle_ids[ghost_cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          for (const auto face : ghost_cell->face_indices())
            {
              if (ghost_cell->at_boundary(face))
                continue;

              const auto add = [&](const auto &ghost_cell,
                                   const auto &local_cell) {
                if (local_cell->is_locally_owned() == false)
                  return;

                const auto neighbor_particle_id =
                  particle_ids[local_cell->global_active_cell_index()];

                if (neighbor_particle_id == invalid_particle_id)
                  return;

                auto &temp =
                  local_connectivity[neighbor_particle_id - local_offset];
                temp.emplace_back(ghost_cell->subdomain_id(), particle_id);
                std::sort(temp.begin(), temp.end());
                temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
              };

              if (ghost_cell->neighbor(face)->has_children())
                {
                  for (unsigned int subface = 0;
                       subface <
                       GeometryInfo<dim>::n_subfaces(
                         dealii::internal::SubfaceCase<dim>::case_isotropic);
                       ++subface)
                    add(ghost_cell,
                        ghost_cell->neighbor_child_on_subface(face, subface));
                }
              else
                add(ghost_cell, ghost_cell->neighbor(face));
            }
        }

    return local_connectivity;
  }

  unsigned int
  number_of_stitched_particles(
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const MPI_Comm                   comm)
  {
    // Determine properties of particles (volume, radius, center)
    unsigned int n_particles = 0;

    // Determine the number of particles
    if (Utilities::MPI::sum(local_to_global_particle_ids.size(), comm) == 0)
      n_particles = 0;
    else
      {
        n_particles = (local_to_global_particle_ids.size() == 0) ?
                        0 :
                        *std::max_element(local_to_global_particle_ids.begin(),
                                          local_to_global_particle_ids.end());
        n_particles = Utilities::MPI::max(n_particles, comm) + 1;
      }

    return n_particles;
  }

  template <int dim, typename VectorIds>
  std::tuple<unsigned int,            // n_particles
             std::vector<Point<dim>>, // particle_centers
             std::vector<double>,     // particle_radii
             std::vector<double>,     // particle_measures
             std::vector<double>>     // particle_max_values
  compute_particles_info(
    const DoFHandler<dim> &          dof_handler,
    const VectorIds &                particle_ids,
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const unsigned int               local_offset,
    const double                     invalid_particle_id       = -1.0,
    const std::vector<double> &      local_particle_max_values = {})
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int n_particles =
      number_of_stitched_particles(local_to_global_particle_ids, comm);

    const unsigned int  n_features = 1 + dim;
    std::vector<double> particle_info(n_particles * n_features);
    std::vector<double> particle_max_values(n_particles);

    // Compute local information
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto particle_id =
            particle_ids[cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          const unsigned int local_id =
            static_cast<unsigned int>(particle_id) - local_offset;
          const unsigned int unique_id = local_to_global_particle_ids[local_id];

          AssertIndexRange(unique_id, n_particles);

          particle_info[n_features * unique_id + 0] += cell->measure();

          for (unsigned int d = 0; d < dim; ++d)
            particle_info[n_features * unique_id + 1 + d] +=
              cell->center()[d] * cell->measure();

          if (!local_particle_max_values.empty())
            particle_max_values[unique_id] =
              local_particle_max_values[local_id];
        }

    // Reduce information - particles info
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_info.data(),
                  particle_info.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    // Reduce information - particles max values
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_max_values.data(),
                  particle_max_values.size(),
                  MPI_DOUBLE,
                  MPI_MAX,
                  comm);

    // Compute particles centers
    std::vector<Point<dim>> particle_centers(n_particles);
    std::vector<double>     particle_measures(n_particles);
    for (unsigned int i = 0; i < n_particles; i++)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            particle_centers[i][d] = particle_info[i * n_features + 1 + d] /
                                     particle_info[i * n_features];
          }
        particle_measures[i] = particle_info[i * n_features];
      }

    // Compute particles radii
    std::vector<double> particle_radii(n_particles, 0.);
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto particle_id =
            particle_ids[cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          const unsigned int local_id =
            static_cast<unsigned int>(particle_id) - local_offset;

          AssertIndexRange(local_id, local_to_global_particle_ids.size());

          const unsigned int unique_id = local_to_global_particle_ids[local_id];

          AssertIndexRange(unique_id, n_particles);

          const auto &center = particle_centers[unique_id];

          const double dist =
            center.distance(cell->barycenter()) + cell->diameter() / 2.;
          particle_radii[unique_id] = std::max(particle_radii[unique_id], dist);
        }

    // Reduce information - particles radii
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_radii.data(),
                  particle_radii.size(),
                  MPI_DOUBLE,
                  MPI_MAX,
                  comm);

    return std::make_tuple(n_particles,
                           std::move(particle_centers),
                           std::move(particle_radii),
                           std::move(particle_measures),
                           std::move(particle_max_values));
  }

  template <int dim, typename VectorIds>
  std::vector<double>
  compute_particles_inertia(
    const DoFHandler<dim> &          dof_handler,
    const VectorIds &                particle_ids,
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const unsigned int               local_offset,
    const std::vector<Point<dim>> &  particle_centers,
    const double                     invalid_particle_id = -1.0)
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int n_particles = particle_centers.size();

    // Compute particles moments of inertia
    std::vector<double> particle_inertia(n_particles * num_inertias<dim>, 0.);
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto particle_id =
            particle_ids[cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          const unsigned int local_id =
            static_cast<unsigned int>(particle_id) - local_offset;

          AssertIndexRange(local_id, local_to_global_particle_ids.size());

          const unsigned int unique_id = local_to_global_particle_ids[local_id];

          AssertIndexRange(unique_id, n_particles);

          const auto &center  = particle_centers[unique_id];
          const auto  r_local = Point<dim>(cell->center() - center);

          evaluate_inertia_properties(
            r_local,
            cell->measure(),
            &(particle_inertia[num_inertias<dim> * unique_id]));
        }

    // Reduce information - particles info
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_inertia.data(),
                  particle_inertia.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    return particle_inertia;
  }

  template <int dim, typename VectorSolution, typename VectorIds>
  std::tuple<unsigned int, std::vector<unsigned int>, std::vector<double>>
  detect_local_particle_groups(VectorIds &            particle_ids,
                               const DoFHandler<dim> &dof_handler,
                               const VectorSolution & solution,
                               const bool     stitching_via_graphs = true,
                               const double   threshold_lower      = 0.01,
                               const double   invalid_particle_id  = -1.0,
                               MyTimerOutput *timer                = nullptr)
  {
    const MPI_Comm comm = dof_handler.get_communicator();

    // step 1) run flooding and determine local particles and give them
    // local ids
    particle_ids = invalid_particle_id;

    unsigned int counter      = 0;
    unsigned int offset       = 0;
    double       op_max_value = std::numeric_limits<double>::lowest();

    std::vector<double> local_particle_max_values;

    {
      ScopedName sc("run_flooding");
      MyScope    scope(sc, timer);

      const bool has_ghost_elements = solution.has_ghost_elements();

      if (has_ghost_elements == false)
        solution.update_ghost_values();

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (run_flooding<dim>(cell,
                                solution,
                                particle_ids,
                                counter,
                                op_max_value,
                                threshold_lower,
                                invalid_particle_id) > 0)
            {
              counter++;
              local_particle_max_values.push_back(op_max_value);
              op_max_value = std::numeric_limits<double>::lowest();
            }
        }

      if (has_ghost_elements == false)
        solution.zero_out_ghost_values();
    }

    // step 2) determine the global number of locally determined particles
    // and give each one an unique id by shifting the ids
    MPI_Exscan(&counter, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

    for (auto &particle_id : particle_ids)
      if (particle_id != invalid_particle_id)
        particle_id += offset;

    // step 3) get particle ids on ghost cells and figure out if local
    // particles and ghost particles might be one particle
    particle_ids.update_ghost_values();

    auto local_connectivity = build_local_connectivity(
      dof_handler, particle_ids, counter, offset, invalid_particle_id);

    // step 4) based on the local-ghost information, figure out all
    // particles on all processes that belong togher (unification ->
    // clique), give each clique an unique id, and return mapping from the
    // global non-unique ids to the global ids
    std::vector<unsigned int> local_to_global_particle_ids;
    {
      ScopedName sc("distributed_stitching");
      MyScope    scope(sc, timer);

      local_to_global_particle_ids =
        stitching_via_graphs ?
          perform_distributed_stitching_via_graph(comm,
                                                  local_connectivity,
                                                  timer) :
          perform_distributed_stitching(comm, local_connectivity, timer);
    }

    return std::make_tuple(offset,
                           std::move(local_to_global_particle_ids),
                           std::move(local_particle_max_values));
  }

  namespace internal
  {
    template <int dim, typename VectorIds>
    void
    run_flooding_prep(
      const typename DoFHandler<dim>::cell_iterator &cell,
      const VectorIds &                              particle_ids,
      VectorIds &                                    particle_markers,
      std::deque<std::vector<
        std::vector<TriaIterator<DoFCellAccessor<dim, dim, false>>>>>
        &                agglomerations,
      const unsigned int aid,
      const unsigned int max_level,
      const double       invalid_particle_id = -1.0)
    {
      if (cell->has_children())
        {
          for (const auto &child : cell->child_iterators())
            run_flooding_prep<dim>(child,
                                   particle_ids,
                                   particle_markers,
                                   agglomerations,
                                   aid,
                                   max_level,
                                   invalid_particle_id);

          return;
        }

      if (!cell->is_locally_owned())
        return;

      const auto particle_id = particle_ids[cell->global_active_cell_index()];
      const auto particle_marker =
        particle_markers[cell->global_active_cell_index()];

      if (particle_id == invalid_particle_id)
        return; // cell outside of the grain group

      if (particle_marker != invalid_particle_id)
        return; // cell has been visited

      particle_markers[cell->global_active_cell_index()] = particle_id;

      for (const auto face : cell->face_indices())
        if (!cell->at_boundary(face))
          {
            const auto neighbor = cell->neighbor(face);

            if (!neighbor->has_children())
              {
                const auto neighbor_particle_id =
                  particle_ids[neighbor->global_active_cell_index()];

                if (neighbor_particle_id == invalid_particle_id)
                  agglomerations[max_level - cell->level()][aid].push_back(
                    cell);
                else
                  run_flooding_prep<dim>(neighbor,
                                         particle_ids,
                                         particle_markers,
                                         agglomerations,
                                         aid,
                                         max_level,
                                         invalid_particle_id);
              }
            else
              {
                for (const auto &child : neighbor->child_iterators())
                  {
                    if (child->is_artificial())
                      continue;

                    const auto child_particle_id =
                      particle_ids[child->global_active_cell_index()];
                    bool do_break = false;

                    if (child_particle_id == invalid_particle_id)
                      {
                        for (const auto &child_of_child :
                             child->child_iterators())
                          {
                            if (child_of_child->is_artificial())
                              continue;

                            if (child_of_child->id() == cell->id())
                              {
                                agglomerations[max_level - cell->level()][aid]
                                  .push_back(cell);
                                do_break = true;
                                break;
                              }
                          }
                      }

                    if (do_break)
                      {
                        break;
                      }
                  }

                run_flooding_prep<dim>(neighbor,
                                       particle_ids,
                                       particle_markers,
                                       agglomerations,
                                       aid,
                                       max_level,
                                       invalid_particle_id);
              }
          }
    }
  } // namespace internal

  template <int dim, typename VectorIds>
  void
  estimate_distances(
    VectorIds &                      particle_distances,
    VectorIds &                      particle_markers,
    const VectorIds &                particle_ids,
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const DoFHandler<dim> &          dof_handler,
    const double                     invalid_particle_id = -1.0,
    MyTimerOutput *                  timer               = nullptr)
  {
    const MPI_Comm comm = dof_handler.get_communicator();

    (void)local_to_global_particle_ids;

    const unsigned int n_global_levels =
      dof_handler.get_triangulation().n_global_levels();
    const unsigned int max_level = n_global_levels - 1;

    const auto h_cell = dof_handler.begin_active(max_level)->diameter();

    std::cout << "h_cell = " << h_cell << std::endl;

    std::deque<
      std::vector<std::vector<TriaIterator<DoFCellAccessor<dim, dim, false>>>>>
      agglomerations(max_level);

    // Set initial
    particle_markers = invalid_particle_id;
    std::cout << "particle_distances.size() = " << particle_distances.size()
              << std::endl;
    std::cout << "particle_markers.size()   = " << particle_markers.size()
              << std::endl;



    // Run preparatory modified flooding
    unsigned int agg_counter = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const auto particle_id = particle_ids[cell->global_active_cell_index()];
        const auto particle_marker =
          particle_markers[cell->global_active_cell_index()];

        if (particle_id == invalid_particle_id ||
            particle_marker != invalid_particle_id)
          continue;

        std::for_each(
          agglomerations.begin(), agglomerations.end(), [](auto &agg_set) {
            agg_set.push_back(
              std::vector<TriaIterator<DoFCellAccessor<dim, dim, false>>>());
          });

        std::cout << "agglomerations.size() = " << agglomerations.size()
                  << std::endl;
        for (const auto &agg_set : agglomerations)
          {
            std::cout << "agg_set.size() = " << agg_set.size() << std::endl;
          }

        internal::run_flooding_prep(cell,
                                    particle_ids,
                                    particle_markers,
                                    agglomerations,
                                    agg_counter,
                                    max_level,
                                    invalid_particle_id);

        ++agg_counter;
      }

    MPI_Barrier(comm);
    std::cout << "agglomerations.size() = " << agglomerations.size()
              << std::endl;


    // Set zero distances
    particle_distances = invalid_particle_id;
    for (const auto &agglomerations_set : agglomerations)
      for (const auto &agglomeration : agglomerations_set)
        for (const auto &cell : agglomeration)
          particle_distances[cell->global_active_cell_index()] =
            std::pow(2, max_level - cell->level()) - 1;

    std::cout << "n_global_levels = " << n_global_levels << std::endl;

    // Now perform loops
    while (std::find_if(agglomerations.cbegin(),
                        agglomerations.cend(),
                        [](const auto &agglomerations_set) {
                          return std::find_if(agglomerations_set.cbegin(),
                                              agglomerations_set.cend(),
                                              [](const auto &agg) {
                                                return !agg.empty();
                                              }) != agglomerations_set.cend();
                        }) != agglomerations.cend())
      {
        // Pick the nearest agglomeration set
        const auto current_agglomerations = agglomerations.front();
        agglomerations.pop_front();

        // Add empty
        std::vector<std::vector<TriaIterator<DoFCellAccessor<dim, dim, false>>>>
          new_agglomerations_set(current_agglomerations.size());
        agglomerations.emplace_back(std::move(new_agglomerations_set));

        agg_counter = 0;
        for (const auto &current_agglomeration : current_agglomerations)
          {
            for (const auto &cell : current_agglomeration)
              {
                const auto cell_particle_id =
                  particle_markers[cell->global_active_cell_index()];
                const auto distance =
                  particle_distances[cell->global_active_cell_index()];

                for (const auto face : cell->face_indices())
                  if (!cell->at_boundary(face))
                    {
                      const auto neighbor = cell->neighbor(face);

                      if (!neighbor->has_children())
                        {
                          if (neighbor->is_locally_owned())
                            {
                              const auto neighbor_particle_id = particle_markers
                                [neighbor->global_active_cell_index()];

                              if (neighbor_particle_id == invalid_particle_id)
                                {
                                  agglomerations[max_level - neighbor->level()]
                                                [agg_counter]
                                                  .push_back(neighbor);
                                  particle_distances
                                    [neighbor->global_active_cell_index()] =
                                      distance +
                                      std::pow(2,
                                               max_level - neighbor->level());
                                  particle_markers
                                    [neighbor->global_active_cell_index()] =
                                      cell_particle_id;
                                }
                            }
                        }
                      else
                        {
                          for (const auto &child : neighbor->child_iterators())
                            {
                              if (child->is_artificial())
                                continue;

                              const auto child_particle_id = particle_markers
                                [child->global_active_cell_index()];

                              if (child_particle_id == invalid_particle_id)
                                {
                                  for (const auto &child_of_child :
                                       child->child_iterators())
                                    {
                                      if (child_of_child->is_artificial())
                                        continue;

                                      if (child_of_child->id() == cell->id())
                                        {
                                          if (child->is_locally_owned())
                                            {
                                              agglomerations[max_level -
                                                             child->level()]
                                                            [agg_counter]
                                                              .push_back(child);
                                              particle_distances
                                                [child
                                                   ->global_active_cell_index()] =
                                                  distance +
                                                  std::pow(2,
                                                           max_level -
                                                             child->level());
                                              particle_markers
                                                [child
                                                   ->global_active_cell_index()] =
                                                  cell_particle_id;
                                            }

                                          break;
                                        }
                                    }
                                }
                            }
                        }
                    }
              }

            // agglomerations.emplace_back(std::move(agglomeration));
            ++agg_counter;
          }

        //++distance;
      }


  }
} // namespace GrainTracker