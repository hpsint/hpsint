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

#include <deal.II/numerics/rtree.h>

#include <pf-applications/base/geometry.h>

#include "grain.h"
#include "output.h"

namespace GrainTracker
{
  using namespace dealii;

  DeclExceptionMsg(ExcGrainsInconsistency, "Grains inconsistency detected!");

  template <int dim>
  std::vector<std::vector<unsigned int>>
  extract_grain_indices_per_op(const std::map<unsigned int, Grain<dim>> &grains,
                               const unsigned int n_order_params)
  {
    std::vector<std::vector<unsigned int>> grains_per_op(n_order_params);

    for (const auto &[grain_id, grain] : grains)
      grains_per_op[grain.get_order_parameter_id()].push_back(grain_id);

    return grains_per_op;
  }

  // Advanced ids assignment algo based on rtrees
  template <int dim>
  std::map<unsigned int, unsigned int>
  transfer_grain_ids(const std::map<unsigned int, Grain<dim>> &new_grains,
                     const std::map<unsigned int, Grain<dim>> &old_grains,
                     const unsigned int                        n_order_params)
  {
    // This algorithm works now only for grains with a single segment
    namespace bgi = boost::geometry::index;

    using ContainerType = std::vector<BoundingBox<dim>>;

    const auto old_grains_per_op =
      extract_grain_indices_per_op(old_grains, n_order_params);
    const auto new_grains_per_op =
      extract_grain_indices_per_op(new_grains, n_order_params);

    // This can be refactored such that vector is used
    std::map<unsigned int, unsigned int> new_grains_to_old;
    std::map<unsigned int, unsigned int> old_grains_to_new;

    for (unsigned int op = 0; op < n_order_params; ++op)
      {
        ContainerType boxes;

        std::vector<unsigned int> old_segments_to_grains;

        for (const unsigned int grain_id : new_grains_per_op[op])
          new_grains_to_old.try_emplace(
            grain_id, std::numeric_limits<unsigned int>::max());

        for (const unsigned int grain_id : old_grains_per_op[op])
          for (const auto &old_segment : old_grains.at(grain_id).get_segments())
            {
              // Flatten all segments of all grains for the given order
              // parameter
              old_segments_to_grains.push_back(grain_id);

              boxes.emplace_back(
                create_bounding_box_around_point(old_segment.get_center(),
                                                 old_segment.get_radius()));
            }

        const auto tree = pack_rtree_of_indices(boxes);

        for (const unsigned int grain_id : new_grains_per_op[op])
          for (const auto &new_segment : new_grains.at(grain_id).get_segments())
            {
              const auto box =
                create_bounding_box_around_point(new_segment.get_center(),
                                                 new_segment.get_radius());

              std::vector<typename ContainerType::size_type> result;

              tree.query(bgi::intersects(box) &&
                           bgi::nearest(new_segment.get_center(), 1),
                         std::back_inserter(result));

              // If any of the grain was identified, then there is no need to
              // check for the others, so we break the iteration
              if (!result.empty())
                {
                  const unsigned int old_grain_id =
                    old_segments_to_grains[result[0]];

                  const auto it_old_to_new =
                    old_grains_to_new.find(old_grain_id);

                  if (it_old_to_new != old_grains_to_new.end())
                    {
                      std::ostringstream ss;
                      ss << "Mapping conflict has been detected." << std::endl;
                      ss << "An old grain is mapped at least to 2 new grains:"
                         << std::endl;
                      ss << std::endl;

                      ss << "old grain:" << std::endl;
                      print_grain(old_grains.at(old_grain_id), ss);
                      ss << std::endl;

                      ss << "new grain 1:" << std::endl;
                      print_grain(new_grains.at(it_old_to_new->second), ss);
                      ss << std::endl;

                      ss << "new grain 2:" << std::endl;
                      print_grain(new_grains.at(grain_id), ss);

                      // Thrown an exception
                      AssertThrow(it_old_to_new == old_grains_to_new.end(),
                                  ExcGrainsInconsistency(ss.str()));
                    }

                  new_grains_to_old.at(grain_id) = old_grain_id;
                  old_grains_to_new.try_emplace(old_grain_id, grain_id);

                  break;
                }
            }
      }

    return new_grains_to_old;
  }

  // Build a set of active order parameters from the map of grains
  template <int dim>
  std::set<unsigned int>
  extract_active_order_parameter_ids(
    const std::map<unsigned int, Grain<dim>> &grains)
  {
    std::set<unsigned int> active_op_ids;

    for (const auto &[gid, gr] : grains)
      {
        (void)gid;
        active_op_ids.insert(gr.get_order_parameter_id());
      }

    return active_op_ids;
  }

} // namespace GrainTracker
