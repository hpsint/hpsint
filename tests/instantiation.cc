// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#define MAX_SINTERING_GRAINS 5

#include <pf-applications/sintering/instantiation.h>

#include <array>
#include <iostream>

template <int c, int d>
void
dump_imp()
{
  std::cout << c << " " << d << std::endl;
}

template <typename U>
struct OperatorDummyLite
{
  using T = OperatorDummyLite;

  void
  dump() const
  {
#define OPERATION(c, d) (std::cout << c << " " << d << std::endl);
    EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
  }

  unsigned int
  n_components() const
  {
    return 5;
  }
};

template <typename U>
struct OperatorDummyExtended
{
  using T = OperatorDummyExtended;

  void
  dump() const
  {
#define OPERATION(c, d) dump_imp<c, d>();
    EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
  }

  unsigned int
  n_components() const
  {
    return 5;
  }

  unsigned int
  n_grains() const
  {
    return 3;
  }

  static constexpr unsigned int
  n_grains_to_n_components(const unsigned int n_grains)
  {
    return n_grains + 2;
  }
};

int
main()
{
  std::cout << std::boolalpha;
  std::cout << "has_n_grains_method<OperatorDummyLite>     = "
            << has_n_grains_method<OperatorDummyLite<double>> << std::endl;
  std::cout << "has_n_grains_method<OperatorDummyExtended> = "
            << has_n_grains_method<OperatorDummyExtended<double>> << std::endl;
  std::cout << std::noboolalpha;

  OperatorDummyLite<double> opl;
  opl.dump();

  OperatorDummyExtended<double> ope;
  ope.dump();
}