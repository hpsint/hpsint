// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;

std::vector<unsigned int>
GrainTracker::connected_components(
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
GrainTracker::perform_distributed_stitching_via_graph(
  const MPI_Comm comm,
  const std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
                &edges_in,
  MyTimerOutput *timer)
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
GrainTracker::perform_distributed_stitching(
  const MPI_Comm                                                   comm,
  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input,
  MyTimerOutput                                                   *timer)
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
        [&](const unsigned int other_rank) { return data_to_send[other_rank]; },
        [&](const unsigned int, const auto &data) {
          for (const auto &data_i : data)
            {
              const unsigned int index   = std::get<0>(data_i) - offset;
              const auto        &values  = std::get<1>(data_i);
              auto              &input_i = input[index];

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
  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input_valid;

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

  std::vector<unsigned int> result(input.size(), numbers::invalid_unsigned_int);

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
      [&](const unsigned int other_rank) { return data_to_send_[other_rank]; },
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

unsigned int
GrainTracker::number_of_stitched_particles(
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
