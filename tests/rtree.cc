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

#include <deal.II/numerics/rtree.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 2;

  using namespace dealii;

  namespace bgi = boost::geometry::index;

  using PP = std::pair<Point<dim>, Point<dim>>;

  // note: boost can not handle circles/balls
  std::vector<BoundingBox<dim>> boxes;
  boxes.emplace_back(PP{{-0.30, -0.30}, {-0.10, -0.10}});
  boxes.emplace_back(PP{{-0.30, +0.30}, {-0.10, +0.40}});
  boxes.emplace_back(PP{{+0.10, -0.30}, {+0.10, +0.30}});

  const auto tree = pack_rtree_of_indices(boxes);

  BoundingBox<dim> box(PP{{-0.25, -0.25}, {+0.25, +0.25}});

  for (const auto &i : tree | bgi::adaptors::queried(bgi::intersects(box)))
    std::cout << "Point p: " << i << std::endl;
}