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

namespace GrainTracker
{
  using namespace dealii;

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

  /* Advanced ids assignment algo based on rtrees. It currently implies that
   * grain consist of a single segment only. This is OK at the moment. */
  template <int dim>
  std::map<unsigned int, unsigned int>
  transfer_grain_ids(const std::map<unsigned int, Grain<dim>> &new_grains,
                     const std::map<unsigned int, Grain<dim>> &old_grains,
                     const unsigned int                        n_order_params)
  {
    // This algorithm works now only for grains with a single segment
    namespace bgi = boost::geometry::index;

    using PP = std::pair<Point<dim>, Point<dim>>;

    using ContainerType = std::vector<BoundingBox<dim>>;

    const auto old_grains_per_op =
      extract_grain_indices_per_op(old_grains, n_order_params);
    const auto new_grains_per_op =
      extract_grain_indices_per_op(new_grains, n_order_params);

    // This can be refactored such that vector is used
    std::map<unsigned int, unsigned int> new_grains_to_old;

    for (unsigned int op = 0; op < n_order_params; ++op)
      {
        ContainerType boxes;

        for (const unsigned int grain_id : new_grains_per_op[op])
          {
            new_grains_to_old.try_emplace(
              grain_id, std::numeric_limits<unsigned int>::max());

            const Point<dim> &center =
              new_grains.at(grain_id).get_segments()[0].get_center();
            const double radius = new_grains.at(grain_id).get_max_radius();

            boxes.emplace_back(
              create_bounding_box_around_point(center, radius));
          }

        const auto tree = pack_rtree_of_indices(boxes);

        for (const unsigned int grain_id : old_grains_per_op[op])
          {
            const Point<dim> &center =
              old_grains.at(grain_id).get_segments()[0].get_center();
            const double radius = old_grains.at(grain_id).get_max_radius();

            const auto box = create_bounding_box_around_point(center, radius);

            std::vector<typename ContainerType::size_type> result;

            tree.query(bgi::intersects(box) && bgi::nearest(center, 1),
                       std::back_inserter(result));

            if (!result.empty())
              new_grains_to_old.at(new_grains_per_op[op][result[0]]) = grain_id;
          }
      }

    return new_grains_to_old;
  }

} // namespace GrainTracker
