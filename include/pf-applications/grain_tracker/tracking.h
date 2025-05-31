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

#include <functional>
#include <optional>
#include <set>

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
  transfer_grain_ids(
    const std::map<unsigned int, Grain<dim>>           &new_grains,
    const std::map<unsigned int, Grain<dim>>           &old_grains,
    const unsigned int                                  n_order_params,
    std::optional<std::reference_wrapper<std::ostream>> logger = {})
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
              tree.query(bgi::intersects(box), std::back_inserter(result));

              // If any of the grain was identified, then there is no need to
              // check for the others, so we break the iteration
              if (!result.empty())
                {
                  /* Perform manual sorting among the candidates, since
                   * bgi::nearest() outputs the closest to the face of the box,
                   * not the box having the closest center. */

                  unsigned int old_grain_id = 0;

                  if (result.size() > 1)
                    {
                      double min_dist = std::numeric_limits<double>::max();

                      for (const auto &id : result)
                        {
                          const unsigned int candidate_grain_id =
                            old_segments_to_grains[id];

                          // Compute the segment id
                          unsigned int work_id = id;
                          for (; work_id > 0 &&
                                 old_segments_to_grains[work_id - 1] ==
                                   old_segments_to_grains[work_id];
                               --work_id)
                            ;
                          const unsigned int segment_id = id - work_id;

                          const auto &candidate_segment =
                            old_grains.at(candidate_grain_id)
                              .get_segments()[segment_id];

                          const auto curent_dist =
                            candidate_segment.get_center().distance(
                              new_segment.get_center());

                          if (curent_dist < min_dist)
                            {
                              old_grain_id = candidate_grain_id;
                              min_dist     = curent_dist;
                            }
                        }
                    }
                  else
                    {
                      old_grain_id = old_segments_to_grains[result[0]];
                    }

                  // Check if mapping already exists
                  const auto it_old_to_new =
                    old_grains_to_new.find(old_grain_id);

                  // Log the mapping conflicts resolution
                  if (logger && it_old_to_new != old_grains_to_new.end())
                    {
                      auto &ss = logger->get();

                      ss << "Mapping conflict \"old -> new\" has been detected"
                         << std::endl;
                      ss << "  existing mapping: " << old_grain_id << " -> "
                         << it_old_to_new->second << std::endl;
                      ss << " candidate mapping: " << old_grain_id << " -> "
                         << grain_id << std::endl;
                      ss << "old grain:" << std::endl;
                      print_grain(old_grains.at(old_grain_id), ss);

                      ss << "new grain already mapped:" << std::endl;
                      print_grain(new_grains.at(it_old_to_new->second), ss);

                      ss << "new grain candidate for mapping:" << std::endl;
                      print_grain(new_grains.at(grain_id), ss);

                      // This check appears below too, it was decided to
                      // separate loggin logic to the independent block for the
                      // code clarity
                      if (new_grains.at(grain_id).get_max_value() >
                          new_grains.at(it_old_to_new->second).get_max_value())
                        ss
                          << "The existing mapping was overwritten with the candidate."
                          << std::endl;
                      else
                        ss
                          << "The existing mapping was kept and the candidate was omitted."
                          << std::endl;
                    }

                  // Create "new -> old" and "old -> new" mappings. The latter
                  // is always checked to ensure that the mapping "new -> old"
                  // is unambiguous, i.e. an old grain can be mapped only to one
                  // new grain.
                  if (it_old_to_new == old_grains_to_new.end())
                    {
                      // Add new mapping if it does not exist yet
                      new_grains_to_old.at(grain_id) = old_grain_id;
                      old_grains_to_new.try_emplace(old_grain_id, grain_id);
                    }
                  else if (new_grains.at(grain_id).get_max_value() >
                           new_grains.at(it_old_to_new->second).get_max_value())
                    {
                      // Overwrite the existing mapping if a newly detected
                      // candidate grain has a larger max_value than the one
                      // mapped previously
                      new_grains_to_old.at(it_old_to_new->second) =
                        numbers::invalid_unsigned_int;
                      new_grains_to_old.at(grain_id)     = old_grain_id;
                      old_grains_to_new.at(old_grain_id) = grain_id;
                    }
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

  // Get barycenter of a particle cloud
  template <int dim>
  Point<dim>
  calc_cloud_barycenter(const std::map<unsigned int, Grain<dim>> &grains)
  {
    Point<dim> static_moment;
    double     volume = 0;

    for (const auto &[gid, grain] : grains)
      {
        (void)gid;

        for (const auto &segment : grain.get_segments())
          {
            static_moment += segment.get_measure() * segment.get_center();
            volume += segment.get_measure();
          }
      }

    return static_moment / volume;
  }

} // namespace GrainTracker
