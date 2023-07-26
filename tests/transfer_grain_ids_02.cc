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
  // Map 3 old grains to 2 new grains: 1 disappeared

  constexpr unsigned int dim = 2;

  std::map<unsigned int, Grain<dim>> old_grains;

  old_grains.try_emplace(4, 4, 0);
  old_grains.at(4).add_segment(Point<dim>(0, 0),
                               2.0,
                               std::pow(2.0, 2) * M_PI,
                               1.0);

  old_grains.try_emplace(2, 2, 0);
  old_grains.at(2).add_segment(Point<dim>(8, 0),
                               3.0,
                               std::pow(3.0, 2) * M_PI,
                               1.0);

  old_grains.try_emplace(7, 7, 0);
  old_grains.at(7).add_segment(Point<dim>(2, -9),
                               1.0,
                               std::pow(1.0, 2) * M_PI,
                               1.0);

  std::map<unsigned int, Grain<dim>> new_grains;

  new_grains.try_emplace(0, 0, 0);
  new_grains.at(0).add_segment(Point<dim>(1, 1),
                               1.7,
                               std::pow(1.7, 2) * M_PI,
                               1.0);

  new_grains.try_emplace(2, 2, 0);
  new_grains.at(2).add_segment(Point<dim>(7, 1),
                               3.2,
                               std::pow(3.2, 2) * M_PI,
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