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

#include <deal.II/base/exceptions.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <iostream>

namespace GrainTracker
{
  using namespace dealii;

  class PeriodicityGraph
  {
  private:
    // Internal graph
    using Graph = boost::adjacency_list<boost::vecS,
                                        boost::vecS,
                                        boost::undirectedS,
                                        unsigned int>;
    // Vertex
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

  public:
    // Add connection between paticles
    void
    add_connection(const unsigned int particle_from,
                   const unsigned int particle_to)
    {
      const auto &vertex_src = vertex(particle_from);
      const auto &vertex_dst = vertex(particle_to);

      boost::add_edge(vertex_src, vertex_dst, graph);
    }

    // Check if graph empty
    bool
    empty() const
    {
      return boost::num_vertices(graph) == 0;
    }

    // Build groups for periodic particles
    unsigned int
    build_groups(std::vector<unsigned int> &particle_groups) const
    {
      if (empty())
        {
          return 0;
        }
      else
        {
          std::vector<unsigned int> components(boost::num_vertices(graph));
          const unsigned int        num_components =
            boost::connected_components(graph, &components[0]);

          for (const auto &[particle_id, graph_data] : particle_id_to_vertex)
            {
              particle_groups[particle_id] = components[graph_data.first];
            }

          return num_components;
        }
    }

  private:
    /* Get vertex from the graph for a given order parameter. If vertex is
     * absent in the graph, then it will be created. The pointer to the vertex
     * is stored in a map for quick access.
     */
    const Vertex &
    vertex(unsigned int particle_id)
    {
      if (particle_id_to_vertex.find(particle_id) ==
          particle_id_to_vertex.end())
        {
          auto v = boost::add_vertex(graph);
          particle_id_to_vertex.emplace(particle_id,
                                        std::make_pair(vertex_numerator, v));
          graph[v] = particle_id;

          ++vertex_numerator;
        }

      return particle_id_to_vertex.at(particle_id).second;
    };

    // Internal boost graph
    Graph graph;

    /* Map storing references to the graph vertices assigned for order
     * parameters
     */
    std::map<unsigned int, std::pair<unsigned int, Vertex>>
      particle_id_to_vertex;

    unsigned int vertex_numerator = 0;
  };
} // namespace GrainTracker