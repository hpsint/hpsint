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

/* Grains sharing an order parameter that get in touch merge into a single
 * grain. The cases below are taken from a simulation of 663 grains, where six
 * such merges happened: three across the periodic boundary and three inside
 * the domain. The check has to report them and must not report anything if
 * grains cross the periodic boundary, vanish or change their size.
 */

#include <pf-applications/grain_tracker/tracking.h>

#include <iostream>

using namespace dealii;
using namespace GrainTracker;

constexpr int dim = 2;

struct SegmentData
{
  Point<dim> center;
  double     radius;
  double     max_value;
};

using Grains = std::map<unsigned int, Grain<dim>>;

void
add_grain(Grains                         &grains,
          const unsigned int              id,
          const unsigned int              order_parameter,
          const std::vector<SegmentData> &segments)
{
  grains.try_emplace(id, id, order_parameter);
  for (const auto &segment : segments)
    grains.at(id).add_segment(segment.center,
                              segment.radius,
                              M_PI * segment.radius * segment.radius,
                              segment.max_value);
}

void
run_case(const std::string &label,
         const Grains      &old_grains,
         const Grains      &new_grains,
         const bool         print_message = false)
{
  const unsigned int n_order_params         = 12;
  const double       threshold_large_grains = 0.5;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  const auto problems = find_lost_or_merged_grains(new_grains,
                                                   old_grains,
                                                   new_grains_to_old,
                                                   threshold_large_grains);

  std::cout << label << std::endl;
  std::cout << "  n_lost   = " << problems.lost.size() << std::endl;
  std::cout << "  n_merged = " << problems.merged.size() << std::endl;

  for (const auto &[new_id, old_ids] : problems.merged)
    {
      std::cout << "  new grain " << new_id << " overlaps old grains";
      for (const unsigned int old_id : old_ids)
        std::cout << " " << old_id;
      std::cout << std::endl;
    }

  if (print_message && !problems.empty())
    {
      std::cout << std::endl;
      std::cout << describe_lost_or_merged_grains(problems,
                                                  new_grains,
                                                  old_grains,
                                                  threshold_large_grains);
    }

  std::cout << std::endl;
}

int
main()
{
  // Two grains collide across the periodic boundary
  {
    Grains old_grains;
    add_grain(old_grains, 13, 1, {{Point<dim>(4.341, 11.820), 5.870, 0.999}});
    add_grain(old_grains, 91, 1, {{Point<dim>(97.185, 9.056), 2.978, 0.908}});

    // Another order parameter is irrelevant
    add_grain(old_grains, 5, 2, {{Point<dim>(4.0, 12.0), 5.0, 1.0}});

    Grains new_grains;
    add_grain(new_grains,
              100,
              1,
              {{Point<dim>(4.362, 11.829), 5.889, 0.998},
               {Point<dim>(97.162, 9.069), 3.058, 0.896}});
    add_grain(new_grains, 101, 2, {{Point<dim>(4.0, 12.0), 5.0, 1.0}});

    run_case("Collision across the periodic boundary: 13 and 91",
             old_grains,
             new_grains,
             true);
  }

  {
    Grains old_grains;
    add_grain(old_grains, 511, 7, {{Point<dim>(3.183, 23.524), 4.021, 1.0}});
    add_grain(old_grains, 553, 7, {{Point<dim>(90.350, 24.410), 9.370, 1.001}});

    Grains new_grains;
    add_grain(new_grains,
              7,
              7,
              {{Point<dim>(3.200, 23.550), 4.200, 0.999},
               {Point<dim>(90.370, 24.430), 9.710, 1.000}});

    run_case("Collision across the periodic boundary: 511 and 553",
             old_grains,
             new_grains);
  }

  // Two grains collide inside the domain
  {
    Grains old_grains;
    add_grain(old_grains,
              181,
              2,
              {{Point<dim>(62.0205, 16.1025), 2.2702, 0.917}});
    add_grain(old_grains,
              182,
              2,
              {{Point<dim>(67.2383, 17.1156), 2.8747, 1.000}});

    Grains new_grains;
    add_grain(new_grains,
              3,
              2,
              {{Point<dim>(65.2249, 16.7437), 5.7119, 0.996}});

    run_case("Collision inside the domain: 181 and 182",
             old_grains,
             new_grains);
  }

  {
    Grains old_grains;
    add_grain(old_grains,
              638,
              4,
              {{Point<dim>(82.3471, 73.9339), 2.1662, 0.847}});
    add_grain(old_grains,
              349,
              4,
              {{Point<dim>(76.8076, 70.7076), 4.0132, 0.998}});

    Grains new_grains;
    add_grain(new_grains,
              3,
              4,
              {{Point<dim>(78.4013, 71.6251), 6.8259, 0.995}});

    run_case("Collision inside the domain: 638 and 349",
             old_grains,
             new_grains);
  }

  {
    Grains old_grains;
    add_grain(old_grains,
              79,
              0,
              {{Point<dim>(63.7152, 46.4152), 2.5516, 0.558}});
    add_grain(old_grains,
              71,
              0,
              {{Point<dim>(69.7136, 45.1932), 3.3554, 0.976}});

    Grains new_grains;
    add_grain(new_grains,
              3,
              0,
              {{Point<dim>(67.6279, 45.6412), 6.4605, 0.970}});

    run_case("Collision inside the domain: 79 and 71", old_grains, new_grains);
  }

  // A small grain vanishes: its maximum value was already small
  {
    Grains old_grains;
    add_grain(old_grains, 1, 0, {{Point<dim>(20.0, 20.0), 5.0, 1.0}});
    add_grain(old_grains, 2, 0, {{Point<dim>(40.0, 20.0), 1.2, 0.08}});

    Grains new_grains;
    add_grain(new_grains, 5, 0, {{Point<dim>(20.1, 20.0), 5.1, 1.0}});

    run_case("A small grain vanishes", old_grains, new_grains);
  }

  // A small grain vanishes next to a larger one, but it was not large
  {
    Grains old_grains;
    add_grain(old_grains, 1, 0, {{Point<dim>(20.0, 20.0), 5.0, 1.0}});
    add_grain(old_grains, 2, 0, {{Point<dim>(26.0, 20.0), 1.2, 0.08}});

    Grains new_grains;
    add_grain(new_grains, 5, 0, {{Point<dim>(20.5, 20.0), 6.8, 1.0}});

    run_case("A small grain is consumed", old_grains, new_grains);
  }

  // A grain crosses the periodic boundary
  {
    Grains old_grains;
    add_grain(old_grains, 284, 4, {{Point<dim>(4.26, 30.55), 4.94, 1.0}});

    Grains new_grains;
    add_grain(new_grains,
              2,
              4,
              {{Point<dim>(4.25, 30.53), 5.10, 1.000},
               {Point<dim>(99.93, 31.39), 0.46, 0.012}});

    run_case("A grain crosses the periodic boundary", old_grains, new_grains);
  }

  // A grain being periodic stays periodic
  {
    Grains old_grains;
    add_grain(old_grains,
              275,
              3,
              {{Point<dim>(1.25, 23.34), 2.90, 0.891},
               {Point<dim>(98.64, 22.87), 3.03, 0.894}});

    Grains new_grains;
    add_grain(new_grains,
              0,
              3,
              {{Point<dim>(1.11, 23.30), 2.80, 0.789},
               {Point<dim>(98.74, 22.90), 3.22, 0.796}});

    run_case("A periodic grain", old_grains, new_grains);
  }

  // Nothing to compare with
  {
    Grains new_grains;
    add_grain(new_grains, 0, 3, {{Point<dim>(1.11, 23.30), 2.80, 0.789}});

    run_case("No old grains", Grains(), new_grains);
  }
}
