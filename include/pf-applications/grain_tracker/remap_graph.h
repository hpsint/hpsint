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
    using Graph = boost::adjacency_list<boost::listS,
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

      const auto edge = boost::edge(vertex_src, vertex_dst, graph);

      if (edge.second == false || graph[edge.first].grain != grain_id)
        {
          boost::add_edge(vertex_src, vertex_dst, EdgeProp{grain_id}, graph);
        }
    }

    // Check if graph has cycles
    bool
    has_cycles() const
    {
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

      bool has_cycle = false;

      CycleDetector vis(has_cycle);
      boost::depth_first_search(
        graph, boost::visitor(vis).vertex_index_map(ipmap).color_map(cpmap));

      return has_cycle;
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

    // Internal boost graph
    Graph graph;

    /* Map storing references to the graph vertices assigned for order
     * parameters
     */
    std::map<unsigned int, Vertex> order_parameter_to_vertex;

    // Special visitor for cycles checking
    struct CycleDetector : public boost::dfs_visitor<>
    {
      CycleDetector(bool &has_cycle)
        : has_cycle(has_cycle)
      {}

      template <typename Edge, typename Graph>
      void
      back_edge(Edge, Graph &)
      {
        has_cycle = true;
      }

    protected:
      bool &has_cycle;
    };
  };
} // namespace GrainTracker