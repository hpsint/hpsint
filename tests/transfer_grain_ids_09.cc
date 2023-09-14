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
  // Map 2 old grains to 2 new grains: multisegments

  constexpr unsigned int dim = 3;

  const auto add_grain = [](std::map<unsigned int, Grain<dim>> &grains,
                            const unsigned int                  grain_id,
                            const double                        x,
                            const double                        y,
                            const double                        z,
                            const double                        r) {
    grains.try_emplace(grain_id, grain_id, 3);
    grains.at(grain_id).add_segment(Point<dim>(x, y, z),
                                    r,
                                    4. / 3. * std::pow(r, 3) * M_PI,
                                    1.0);
  };

  /* Old grains
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 44
      segment: center = 53.2446 111.374 62.851 | radius = 105.014 | max_value = 1.00019
  op_index_current = 3 | op_index_old = 2 | segments = 1 | grain_index = 54
      segment: center = 158.329 35.8013 165.073 | radius = 48.4995 | max_value = 1.00005
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 59
      segment: center = 78.3938 268.394 57.3512 | radius = 42.231 | max_value = 1.00001
  op_index_current = 3 | op_index_old = 6 | segments = 1 | grain_index = 60
      segment: center = 206.594 255.865 102.191 | radius = 69.555 | max_value = 1.00012
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 61
      segment: center = 227.119 282.278 247.636 | radius = 46.1484 | max_value = 1.00004
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 78
      segment: center = 91.7484 129.185 276.194 | radius = 52.248 | max_value = 1.00009
  op_index_current = 3 | op_index_old = 5 | segments = 1 | grain_index = 92
      segment: center = 274.079 82.588 84.6985 | radius = 49.2785 | max_value = 1.0001
  */

  std::map<unsigned int, Grain<dim>> old_grains;

  add_grain(old_grains, 44, 53.2446, 111.374, 62.851, 105.014);
  add_grain(old_grains, 54, 158.329, 35.8013, 165.073, 48.4995);
  add_grain(old_grains, 59, 78.3938, 268.394, 57.3512, 42.231);
  add_grain(old_grains, 60, 206.594, 255.865, 102.191, 69.555);
  add_grain(old_grains, 61, 227.119, 282.278, 247.636, 46.1484);
  add_grain(old_grains, 78, 91.7484, 129.185, 276.194, 52.248);
  add_grain(old_grains, 92, 274.079, 82.588, 84.6985, 49.2785);

  /* New grains
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 4294967295
      segment: center = 158.376 35.3881 165.533 | radius = 47.3832 | max_value = 0.999985
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 54
      segment: center = 53.4247 111.307 62.984 | radius = 104.959 | max_value = 1.00018
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 59
      segment: center = 78.4738 268.572 57.3432 | radius = 42.2616 | max_value = 1.00001
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 60
      segment: center = 206.329 255.794 102.117 | radius = 69.3212 | max_value = 1.00013
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 61
      segment: center = 227.098 282.343 247.664 | radius = 46.1728 | max_value = 1.00008
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 78
      segment: center = 92.3623 128.483 275.971 | radius = 53.7383 | max_value = 1.00009
  op_index_current = 3 | op_index_old = 3 | segments = 1 | grain_index = 92
      segment: center = 274.28 82.5818 84.719 | radius = 49.3169 | max_value = 1.00008
  */

  std::map<unsigned int, Grain<dim>> new_grains;

  // This grain is unmapped in the original version, should be 54
  add_grain(new_grains, 744, 158.376, 35.3881, 165.533, 47.3832);

  // And this grain should be assigned to 44
  add_grain(new_grains, 754, 53.4247, 111.307, 62.984, 104.959);
  add_grain(new_grains, 759, 78.4738, 268.572, 57.3432, 42.2616);
  add_grain(new_grains, 760, 206.329, 255.794, 102.117, 69.3212);
  add_grain(new_grains, 761, 227.098, 282.343, 247.664, 46.1728);
  add_grain(new_grains, 778, 92.3623, 128.483, 275.971, 53.7383);
  add_grain(new_grains, 792, 274.28, 82.5818, 84.719, 49.3169);

  const unsigned int n_order_params = 4;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;
}