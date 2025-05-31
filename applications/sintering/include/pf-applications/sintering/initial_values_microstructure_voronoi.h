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

#include <deal.II/base/bounding_box.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <pf-applications/sintering/initial_values_microstructure.h>

namespace Sintering
{
  using namespace dealii;

  /* This class reads microstructure data in the format from the phase-field
   * book of Biner https://link.springer.com/book/10.1007/978-3-319-41196-5
   */
  class InitialValuesMicrostructureVoronoi : public InitialValuesMicrostructure
  {
  public:
    InitialValuesMicrostructureVoronoi(
      std::istream            &stream,
      const double             interface_width     = 0.,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset  = 2,
      const bool               concentration_as_void = false,
      const bool               is_accumulative       = false)
      : InitialValuesMicrostructure(interface_width,
                                    interface_direction,
                                    op_components_offset,
                                    concentration_as_void,
                                    is_accumulative)
    {
      constexpr unsigned int dim = 2;

      unsigned int n_points                   = 0;
      unsigned int n_max_vertices_per_segment = 0;
      unsigned int n_segments                 = 0;

      stream >> n_points >> n_max_vertices_per_segment >> n_segments;

      // Read all points
      std::map<unsigned int, Point<dim>> points;
      for (unsigned int ip = 0; ip < n_points; ++ip)
        {
          unsigned int id_point = 0;
          stream >> id_point;

          Point<dim> pt;

          double coord = 0;
          for (unsigned int i = 0; i < dim; i++)
            {
              stream >> coord;
              pt[i] = coord;
            }

          points.try_emplace(id_point, std::move(pt));
        }

      // Grain indices sitting at a vertex
      std::unordered_map<unsigned int, std::set<unsigned int>> connectivity;

      // Read all segments
      std::unordered_map<unsigned int, MicroGrain> initial_grains;
      for (unsigned int ig = 0; ig < n_segments; ig++)
        {
          unsigned int id_segment = 0;
          stream >> id_segment;

          std::vector<Point<2>>     vertices;
          std::vector<unsigned int> vertex_indices;
          unsigned int              id_point = 0;
          for (unsigned int i = 0; i < n_max_vertices_per_segment; i++)
            {
              stream >> id_point;

              if (id_point > 0)
                {
                  vertices.push_back(points.at(id_point));
                  vertex_indices.push_back(id_point);
                }
            }

          // Create segment
          MicroSegment segment(std::make_move_iterator(vertices.begin()),
                               std::make_move_iterator(vertices.end()),
                               interface_width,
                               interface_direction);

          unsigned int id_grain = 0;
          stream >> id_grain;

          for (const auto pt_id : vertex_indices)
            connectivity[pt_id].insert(id_grain);

          auto it_grain = initial_grains.find(id_grain);
          if (it_grain == initial_grains.end())
            {
              const auto [it, res] = initial_grains.try_emplace(id_grain);
              it_grain             = it;
            }

          it_grain->second.add_segment(std::move(segment));
        }

      // Convert to the internal data structures
      std::unordered_map<unsigned int, unsigned int> old_to_new_indices;
      for (auto &[grain_id, grain] : initial_grains)
        {
          grains.push_back(grain);
          old_to_new_indices.try_emplace(grain_id, grains.size() - 1);
        }

      // DSP for colorization if order parameters are compressed
      const auto             n_grains = grains.size();
      DynamicSparsityPattern dsp(n_grains);
      for (const auto &[point_id, grain_ids] : connectivity)
        for (const auto grain_id_i : grain_ids)
          for (const auto grain_id_j : grain_ids)
            if (grain_id_i != grain_id_j)
              {
                const auto new_id_i = old_to_new_indices[grain_id_i];
                const auto new_id_j = old_to_new_indices[grain_id_j];
                dsp.add(new_id_i, new_id_j);
                dsp.add(new_id_j, new_id_i);
              }

      SparsityPattern sp;
      sp.copy_from(dsp);

      std::vector<unsigned int> color_indices(n_grains);
      SparsityTools::color_sparsity_pattern(sp, color_indices);

      for (unsigned int i = 0; i < n_grains; i++)
        order_parameter_to_grains[color_indices[i] - 1].push_back(i);
    }
  };
} // namespace Sintering
