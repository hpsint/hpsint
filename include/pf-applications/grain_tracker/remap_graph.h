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

#include "remapping.h"

namespace GrainTracker
{
  using namespace dealii;

  class RemapGraph
  {
  private:
    // Vertex properties
    struct VertexProp
    {
      unsigned int order_parameter;
    };

    // Edge properties
    struct EdgeProp
    {
      unsigned int grain;
    };

    // Internal graph
    using Graph = boost::adjacency_list<boost::multisetS,
                                        boost::listS,
                                        boost::bidirectionalS,
                                        VertexProp,
                                        EdgeProp>;
    // Vertex
    using Vertex = boost::graph_traits<Graph>::vertex_descriptor;

    // Edge
    using Edge = boost::graph_traits<Graph>::edge_descriptor;

  public:
    /* Add remapping to the graph providing the order parameter ids which we
     * move the grain from and to, respectively.
     */
    void
    add_remapping(const unsigned int order_parameter_id_from,
                  const unsigned int order_parameter_id_to,
                  const unsigned int grain_id)
    {
      const auto &vertex_src = vertex(order_parameter_id_from);
      const auto &vertex_dst = vertex(order_parameter_id_to);

      if (!remapping_exists(vertex_src, vertex_dst, grain_id))
        {
          boost::add_edge(vertex_src, vertex_dst, EdgeProp{grain_id}, graph);
        }
    }

    /* Resolve cyclic remappings. Any remapping cycles if performed as is, will
     * result in damaging either of the grains involved. To resolve it, a
     * temporary vector has to be used. In this case a pair of remappings is
     * returned. The first remaps a grain to a temporary vector, an then the
     * second remaps from the temporary vector to a new order parameter.
     */
    std::vector<std::pair<Remapping, Remapping>>
    resolve_cycles(std::list<Remapping> &remappings)
    {
      std::vector<std::pair<Remapping, Remapping>> remappings_via_temp;

      // Create index map
      std::map<Vertex, size_t> i_map;
      for (auto v : boost::make_iterator_range(boost::vertices(graph)))
        {
          i_map.emplace(v, i_map.size());
        }

      auto ipmap = boost::make_assoc_property_map(i_map);

      // Create color map
      std::vector<boost::default_color_type> c_map(boost::num_vertices(graph));
      auto cpmap = boost::make_iterator_property_map(c_map.begin(), ipmap);

      std::vector<Edge> cycle_edges;

      CycleDetector vis(cycle_edges);
      boost::depth_first_search(
        graph, boost::visitor(vis).vertex_index_map(ipmap).color_map(cpmap));

      for (const auto &e : cycle_edges)
        {
          auto source = boost::source(e, graph);
          auto target = boost::target(e, graph);

          const unsigned int from     = graph[source].order_parameter;
          const unsigned int to       = graph[target].order_parameter;
          const unsigned int grain_id = graph[e].grain;

          Remapping r{grain_id, from, to};

          auto it_split = std::find(remappings.begin(), remappings.end(), r);

          AssertThrow(
            it_split != remappings.end(),
            ExcMessage(
              "Inconsistency between graph and remappings list detected!"));

          Remapping r1{grain_id, from, numbers::invalid_unsigned_int};
          Remapping r2{grain_id, numbers::invalid_unsigned_int, to};
          remappings_via_temp.push_back(std::make_pair(r1, r2));

          remappings.erase(it_split);
          boost::remove_edge(e, graph);
        }

      return remappings_via_temp;
    }

    // Check if graph empty
    bool
    empty() const
    {
      return boost::num_vertices(graph) == 0;
    }

    // Print graph
    template <typename Stream>
    void
    print(Stream &out) const
    {
      out << "Remappings: " << std::endl;
      boost::graph_traits<Graph>::edge_iterator ei, ei_end;
      for (std::tie(ei, ei_end) = boost::edges(graph); ei != ei_end; ++ei)
        {
          auto source = boost::source(*ei, graph);
          auto target = boost::target(*ei, graph);

          out << graph[*ei].grain << " (" << graph[source].order_parameter
              << " -> " << graph[target].order_parameter << ")" << std::endl;
        }
    }

    /* Rearrange remappings. The function checks the remappings list for
     * possible dependencies between them. If such a dependency has been
     * detected, meaning that there is a corresponding edge in the graph, then
     * the remappings have to be properly ordered.
     */
    void
    rearrange(std::list<Remapping> &remappings)
    {
      // Copy remapping list
      std::list<Remapping> old_remappings(remappings);
      remappings.clear();

      boost::graph_traits<Graph>::vertex_iterator   vi, v_end;
      boost::graph_traits<Graph>::in_edge_iterator  iei, iedge_end;
      boost::graph_traits<Graph>::out_edge_iterator oei, oedge_end;

      boost::tie(vi, v_end) = boost::vertices(graph);
      while (vi != v_end)
        {
          boost::tie(oei, oedge_end) = boost::out_edges(*vi, graph);

          if (oei == oedge_end)
            {
              bool rearranged = false;

              for (boost::tie(iei, iedge_end) = boost::in_edges(*vi, graph);
                   iei != iedge_end;
                   ++iei)
                {
                  auto source = boost::source(*iei, graph);
                  auto target = boost::target(*iei, graph);

                  Remapping r{graph[*iei].grain,
                              graph[source].order_parameter,
                              graph[target].order_parameter};

                  auto it_remap =
                    std::find(old_remappings.begin(), old_remappings.end(), r);

                  AssertThrow(
                    it_remap != old_remappings.end(),
                    ExcMessage(
                      "Inconsistency between graph and remappings list detected!"));

                  // Append remapping to the new list as one of the first
                  remappings.push_back(*it_remap);

                  // And remove this remapping from the old list
                  old_remappings.erase(it_remap);

                  rearranged = true;
                }

              if (rearranged)
                {
                  // Remove vertex and delete all edges
                  boost::clear_vertex(*vi, graph);
                  boost::remove_vertex(*vi, graph);

                  // Reinitialize iterators
                  boost::tie(vi, v_end) = boost::vertices(graph);
                }
              else
                {
                  ++vi;
                }
            }
          else
            {
              ++vi;
            }
        }

      // Append the remaining old remappings
      remappings.insert(remappings.end(),
                        old_remappings.begin(),
                        old_remappings.end());
    }

  private:
    /* Get vertex from the graph for a given order parameter. If vertex is
     * absent in the graph, then it will be created. The pointer to the vertex
     * is stored in a map for quick access.
     */
    const Vertex &
    vertex(unsigned int order_parameter_id)
    {
      if (order_parameter_to_vertex.find(order_parameter_id) ==
          order_parameter_to_vertex.end())
        {
          auto v = boost::add_vertex(graph);
          order_parameter_to_vertex.emplace(order_parameter_id, v);
          graph[v].order_parameter = order_parameter_id;
        }

      return order_parameter_to_vertex.at(order_parameter_id);
    };

    /* Check if a given remapping exists in the graph. */
    bool
    remapping_exists(const Vertex      &vertex_src,
                     const Vertex      &vertex_dst,
                     const unsigned int grain_id) const
    {
      for (auto e : boost::make_iterator_range(
             boost::edge_range(vertex_src, vertex_dst, graph)))
        if (graph[e].grain == grain_id)
          return true;

      return false;
    }

    // Internal boost graph
    Graph graph;

    /* Map storing references to the graph vertices assigned for order
     * parameters
     */
    std::map<unsigned int, Vertex> order_parameter_to_vertex;

    // Special visitor for cycles checking
    struct CycleDetector : public boost::dfs_visitor<>
    {
      CycleDetector(std::vector<Edge> &edges)
        : edges(edges)
      {}

      template <typename Edge, typename Graph>
      void
      back_edge(Edge e, Graph &)
      {
        edges.push_back(e);
      }

    protected:
      std::vector<Edge> &edges;
    };
  };
} // namespace GrainTracker