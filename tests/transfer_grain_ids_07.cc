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
    grains.try_emplace(grain_id, grain_id, 0);
    grains.at(grain_id).add_segment(Point<dim>(x, y, z),
                                    r,
                                    4. / 3. * std::pow(r, 3) * M_PI,
                                    1.0);
  };

  /* Old grains
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 0
      segment: center = 94.6154 29.1288 29.5799 | radius = 45.9676 | max_value = 0.99966
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 1
      segment: center = 220.696 45.5308 34.6221 | radius = 55.8711 | max_value = 1.00013
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 4
      segment: center = 33.2217 247.147 31.5111 | radius = 76.1837 | max_value = 1.00012
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 7
      segment: center = 64.537 59.9542 217.464 | radius = 118.745 | max_value = 1.0003
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 12
      segment: center = 235.693 180.914 188.382 | radius = 64.1163 | max_value = 0.999973
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 15
      segment: center = 69.2122 275.918 193.333 | radius = 51.0762 | max_value = 1.0001
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 19
      segment: center = 204.213 262.26 289.578 | radius = 41.3436 | max_value = 0.999883
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 38
      segment: center = 252.499 278.669 30.8942 | radius = 44.5147 | max_value = 1.00002
  op_index_current = 0 | op_index_old = 2 | segments = 1 | grain_index = 64
      segment: center = 17.671 75.6835 17.7423 | radius = 34.5737 | max_value = 0.969351
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 69
      segment: center = 152.59 179.064 28.2257 | radius = 37.6444 | max_value = 0.999528
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 74
      segment: center = 275.344 27.2085 262.943 | radius = 46.8655 | max_value = 0.999831
  */

  std::map<unsigned int, Grain<dim>> old_grains;

  add_grain(old_grains, 0, 94.6154, 29.1288, 29.5799, 45.9676);
  add_grain(old_grains, 1, 220.696, 45.5308, 34.6221, 55.8711);
  add_grain(old_grains, 4, 33.2217, 247.147, 31.5111, 76.1837);
  add_grain(old_grains, 7, 64.537, 59.9542, 217.464, 118.745);
  add_grain(old_grains, 12, 235.693, 180.914, 188.382, 64.1163);
  add_grain(old_grains, 15, 69.2122, 275.918, 193.333, 51.0762);
  add_grain(old_grains, 19, 204.213, 262.26, 289.578, 41.3436);
  add_grain(old_grains, 38, 252.499, 278.669, 30.8942, 44.5147);
  add_grain(old_grains, 64, 17.671, 75.6835, 17.7423, 34.5737);
  add_grain(old_grains, 69, 152.59, 179.064, 28.2257, 37.6444);
  add_grain(old_grains, 74, 275.344, 27.2085, 262.943, 46.8655);

  /* New grains
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 1
      segment: center = 220.84 45.518 34.6399 | radius = 56.0837 | max_value = 1.00014
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 4
      segment: center = 33.1791 247.382 31.4546 | radius = 75.9383 | max_value = 1.00013
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 7
      segment: center = 64.6469 59.9564 217.528 | radius = 118.845 | max_value = 1.00026
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 12
      segment: center = 235.507 181.562 188.514 | radius = 62.313 | max_value = 0.99996
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 15
      segment: center = 68.6011 275.475 194.369 | radius = 55.9826 | max_value = 1.0001
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 19
      segment: center = 204.239 262.288 289.49 | radius = 41.4039 | max_value = 0.999883
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 38
      segment: center = 252.517 278.719 30.8824 | radius = 44.5405 | max_value = 0.999993
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 64
      segment: center = 94.2105 26.5377 27.7914 | radius = 43.5742 | max_value = 0.999038
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 69
      segment: center = 152.433 179.354 28.2189 | radius = 37.3907 | max_value = 0.999463
  op_index_current = 0 | op_index_old = 0 | segments = 1 | grain_index = 74
      segment: center = 275.57 26.7379 263.176 | radius = 46.587 | max_value = 0.999785
  */

  std::map<unsigned int, Grain<dim>> new_grains;

  add_grain(new_grains, 701, 220.84, 45.518, 34.6399, 56.0837);
  add_grain(new_grains, 704, 33.1791, 247.382, 31.4546, 75.9383);
  add_grain(new_grains, 707, 64.6469, 59.9564, 217.528, 118.845);
  add_grain(new_grains, 712, 235.507, 181.562, 188.514, 62.313);
  add_grain(new_grains, 715, 68.6011, 275.475, 194.369, 55.9826);
  add_grain(new_grains, 719, 204.239, 262.288, 289.49, 41.4039);
  add_grain(new_grains, 738, 252.517, 278.719, 30.8824, 44.5405);

  // This grain should be assigned to grain_id=0
  add_grain(new_grains, 764, 94.2105, 26.5377, 27.7914, 43.5742);

  add_grain(new_grains, 769, 152.433, 179.354, 28.2189, 37.3907);
  add_grain(new_grains, 774, 275.57, 26.7379, 263.176, 46.587);

  const unsigned int n_order_params = 1;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;
}