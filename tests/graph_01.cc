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

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;

int
main()
{
  const unsigned int N = 9;

  std::vector<std::tuple<unsigned int, unsigned int>> edges;

  edges.emplace_back(0, 1);
  edges.emplace_back(0, 4);
  edges.emplace_back(1, 7);
  edges.emplace_back(4, 7);
  edges.emplace_back(2, 5);
  edges.emplace_back(3, 8);

  const auto c = GrainTracker::connected_components(N, edges);

  for (const auto i : c)
    std::cout << i << " ";
  std::cout << std::endl;
}