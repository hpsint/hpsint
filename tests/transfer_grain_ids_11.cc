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

#include <deal.II/base/mpi.h>

#include <pf-applications/grain_tracker/tracking.h>

using namespace dealii;
using namespace GrainTracker;

int
main()
{
  // Map 3 old grains to 5 new grains: 2 new appeared, a conflict between the
  // new grains

  constexpr unsigned int dim = 2;

  std::map<unsigned int, Grain<dim>> old_grains;

  old_grains.try_emplace(14, 14, 0);
  old_grains.at(14).add_segment(Point<dim>(0, 0),
                                2.0,
                                std::pow(2.0, 2) * M_PI,
                                1.0);

  old_grains.try_emplace(12, 12, 0);
  old_grains.at(12).add_segment(Point<dim>(8, 0),
                                3.0,
                                std::pow(3.0, 2) * M_PI,
                                1.0);

  old_grains.try_emplace(17, 17, 0);
  old_grains.at(17).add_segment(Point<dim>(2, -9),
                                1.0,
                                std::pow(1.0, 2) * M_PI,
                                1.0);

  std::map<unsigned int, Grain<dim>> new_grains;

  // This grain is in conflict with grain 4 to be mapped to grain 14
  new_grains.try_emplace(0, 0, 0);
  new_grains.at(0).add_segment(Point<dim>(1.9, 1.9),
                               0.12,
                               std::pow(0.12, 2) * M_PI,
                               0.015);

  // This should be mapped to 12
  new_grains.try_emplace(2, 2, 0);
  new_grains.at(2).add_segment(Point<dim>(7, 1),
                               3.2,
                               std::pow(3.2, 2) * M_PI,
                               1.0);

  // This should be mapped to 17
  new_grains.try_emplace(1, 1, 0);
  new_grains.at(1).add_segment(Point<dim>(3, -8),
                               1.2,
                               std::pow(1.2, 2) * M_PI,
                               1.0);

  // This is a valid new grain
  new_grains.try_emplace(3, 3, 0);
  new_grains.at(3).add_segment(Point<dim>(7, -6),
                               0.9,
                               std::pow(0.9, 2) * M_PI,
                               1.0);

  // This should be mapped to 14
  new_grains.try_emplace(4, 4, 0);
  new_grains.at(4).add_segment(Point<dim>(1, 1),
                               1.7,
                               std::pow(1.7, 2) * M_PI,
                               1.0);

  const unsigned int n_order_params = 1;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;
}