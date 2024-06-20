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

#include <pf-applications/grain_tracker/output.h>

// include input and output archivers
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// include this header to serialize vectors
#include <boost/serialization/vector.hpp>

using namespace dealii;
using namespace GrainTracker;

int
main()
{
  constexpr unsigned int dim = 2;

  std::vector<Grain<dim>> old_grains;
  std::vector<Grain<dim>> new_grains;

  // A grain with one segment
  Grain<dim> g1(4, 7, 14);
  g1.add_segment(Point<dim>(0, 0), 2.0, std::pow(2.0, 2) * M_PI, 1.0);
  old_grains.push_back(g1);

  // A grain containing 2 segments: small + large
  Grain<dim> g2(2, 3, 12);
  g2.add_segment(Point<dim>(2, -9), 1.0, std::pow(1.0, 2) * M_PI, 1.0);
  g2.add_segment(Point<dim>(8, 0), 3.0, std::pow(3.0, 2) * M_PI, 1.0);
  old_grains.push_back(g2);

  // Write
  std::stringstream             sstream;
  boost::archive::text_oarchive fosb(sstream);
  fosb << old_grains;

  // Read
  boost::archive::text_iarchive fisb(sstream);
  fisb >> new_grains;

  std::cout << "Original grains:" << std::endl;
  for (const auto &grain : old_grains)
    print_grain(grain, std::cout);
  std::cout << std::endl;

  std::cout << "Loaded grains:" << std::endl;
  for (const auto &grain : new_grains)
    print_grain(grain, std::cout);
}