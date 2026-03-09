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
  // Map 3 old grains to 3 new grains: a trick here is that one of the new
  // grains starts having a tiny second segment that overlaps also another old
  // grain.

  constexpr unsigned int dim = 2;

  std::map<unsigned int, Grain<dim>> old_grains;

  old_grains.try_emplace(12, 12, 0);
  old_grains.at(12).add_segment(Point<dim>(2.11832, 26.5732),
                                6.5099,
                                std::pow(6.5099, 2) * M_PI,
                                0.999958);
  old_grains.at(12).add_segment(Point<dim>(28.5559, 24.4199),
                                7.56384,
                                std::pow(7.56384, 2) * M_PI,
                                0.999985);

  old_grains.try_emplace(13, 13, 0);
  old_grains.at(13).add_segment(Point<dim>(20.3557, 8.60312),
                                8.97203,
                                std::pow(8.97203, 2) * M_PI,
                                1.00001);

  old_grains.try_emplace(18, 18, 0);
  old_grains.at(18).add_segment(Point<dim>(6.2514, 10.416),
                                4.67711,
                                std::pow(4.67711, 2) * M_PI,
                                0.9953);

  std::map<unsigned int, Grain<dim>> new_grains;

  // This should be mapped to 13, but now has 2 segments
  new_grains.try_emplace(0, 0, 0);
  new_grains.at(0).add_segment(Point<dim>(20.3172, 8.57612),
                               8.99759,
                               std::pow(8.99759, 2) * M_PI,
                               1.00001);
  new_grains.at(0).add_segment(Point<dim>(22.4286, 31.9286),
                               0.172444,
                               std::pow(0.172444, 2) * M_PI,
                               0.0101435);

  // This should be mapped to 12
  new_grains.try_emplace(1, 1, 0);
  new_grains.at(1).add_segment(Point<dim>(2.05773, 26.4717),
                               6.39441,
                               std::pow(6.39441, 2) * M_PI,
                               0.999964);
  new_grains.at(1).add_segment(Point<dim>(28.5804, 24.4466),
                               7.52899,
                               std::pow(7.52899, 2) * M_PI,
                               0.999991);

  // This should be mapped to 18
  new_grains.try_emplace(2, 2, 0);
  new_grains.at(2).add_segment(Point<dim>(6.26648, 10.4237),
                               4.50863,
                               std::pow(4.50863, 2) * M_PI,
                               0.994698);

  const unsigned int n_order_params = 1;

  std::stringstream ss;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params, ss);

  std::cout << ss.str();

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;
}