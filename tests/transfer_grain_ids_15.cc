// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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
  // Order parameter 3 grains taken from a failing track() call. New grain 23
  // has a main segment overlapping old grain 23 and a tiny periodic sliver
  // overlapping old grain 22, which is claimed by new grain 22. The grain 23
  // must be mapped to old grain 23, but the sliver overrides the assignment.

  constexpr unsigned int dim = 2;

  std::map<unsigned int, Grain<dim>> old_grains;

  old_grains.try_emplace(22, 22, 3);
  old_grains.at(22).add_segment(Point<dim>(86.1776, 58.4871),
                                12.9179,
                                std::pow(12.9179, 2) * M_PI,
                                1.01257);

  old_grains.try_emplace(23, 23, 3);
  old_grains.at(23).add_segment(Point<dim>(12.1093, 75.361),
                                14.7067,
                                std::pow(14.7067, 2) * M_PI,
                                1.01264);

  old_grains.try_emplace(25, 25, 3);
  old_grains.at(25).add_segment(Point<dim>(63.7327, 47.5246),
                                9.97812,
                                std::pow(9.97812, 2) * M_PI,
                                1.0108);

  old_grains.try_emplace(26, 26, 3);
  old_grains.at(26).add_segment(Point<dim>(95.307, 8.76812),
                                11.814,
                                std::pow(11.814, 2) * M_PI,
                                1.0196);
  old_grains.at(26).add_segment(Point<dim>(3.84446, 10.0031),
                                10.7252,
                                std::pow(10.7252, 2) * M_PI,
                                1.01137);
  old_grains.at(26).add_segment(Point<dim>(0.388934, 99.7531),
                                0.662999,
                                std::pow(0.662999, 2) * M_PI,
                                0.549259);
  old_grains.at(26).add_segment(Point<dim>(98.3984, 99.6295),
                                2.70646,
                                std::pow(2.70646, 2) * M_PI,
                                0.786676);

  old_grains.try_emplace(29, 29, 3);
  old_grains.at(29).add_segment(Point<dim>(21.9546, 38.7173),
                                14.6635,
                                std::pow(14.6635, 2) * M_PI,
                                1.01179);

  old_grains.try_emplace(33, 33, 3);
  old_grains.at(33).add_segment(Point<dim>(49.1231, 17.4311),
                                10.1754,
                                std::pow(10.1754, 2) * M_PI,
                                1.01375);

  old_grains.try_emplace(34, 34, 3);
  old_grains.at(34).add_segment(Point<dim>(43.0854, 81.935),
                                15.6642,
                                std::pow(15.6642, 2) * M_PI,
                                1.01534);

  std::map<unsigned int, Grain<dim>> new_grains;

  new_grains.try_emplace(22, 22, 3);
  new_grains.at(22).add_segment(Point<dim>(86.1876, 58.4787),
                                13.3329,
                                std::pow(13.3329, 2) * M_PI,
                                1.00226);

  // Must be mapped to old grain 23
  new_grains.try_emplace(23, 23, 3);
  new_grains.at(23).add_segment(Point<dim>(12.187, 75.3819),
                                15.1788,
                                std::pow(15.1788, 2) * M_PI,
                                1.00251);
  new_grains.at(23).add_segment(Point<dim>(99.8648, 70.5392),
                                4.2066,
                                std::pow(4.2066, 2) * M_PI,
                                0.0397948);

  new_grains.try_emplace(25, 25, 3);
  new_grains.at(25).add_segment(Point<dim>(63.6849, 47.5431),
                                10.4076,
                                std::pow(10.4076, 2) * M_PI,
                                1.00236);

  new_grains.try_emplace(26, 26, 3);
  new_grains.at(26).add_segment(Point<dim>(95.014, 8.89503),
                                12.3355,
                                std::pow(12.3355, 2) * M_PI,
                                1.00275);
  new_grains.at(26).add_segment(Point<dim>(4.11595, 10.0784),
                                10.8944,
                                std::pow(10.8944, 2) * M_PI,
                                1.00215);
  new_grains.at(26).add_segment(Point<dim>(97.7288, 99.4498),
                                4.17853,
                                std::pow(4.17853, 2) * M_PI,
                                0.803105);
  new_grains.at(26).add_segment(Point<dim>(0.697694, 99.5613),
                                1.37456,
                                std::pow(1.37456, 2) * M_PI,
                                0.669717);

  new_grains.try_emplace(29, 29, 3);
  new_grains.at(29).add_segment(Point<dim>(22.021, 38.7151),
                                15.1458,
                                std::pow(15.1458, 2) * M_PI,
                                1.00316);

  new_grains.try_emplace(33, 33, 3);
  new_grains.at(33).add_segment(Point<dim>(49.0999, 17.4307),
                                10.6143,
                                std::pow(10.6143, 2) * M_PI,
                                1.00233);

  new_grains.try_emplace(34, 34, 3);
  new_grains.at(34).add_segment(Point<dim>(43.1081, 81.9711),
                                16.1218,
                                std::pow(16.1218, 2) * M_PI,
                                1.00224);

  const unsigned int n_order_params = 4;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;

  // Mirrors the check in Tracker::track() that aborts the run
  AssertThrow(new_grains_to_old.at(23) == 23,
              ExcMessage("New grain 23 was not mapped to old grain 23"));
}
