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

#include <pf-applications/tests/instantiation_tester.h>

int
main()
{
  Test::Variants variants_ch_ac = {{1, false},
                                   {0, false},
                                   {MAX_SINTERING_GRAINS + 1, true}};
  Test::run_instantiation<Test::Expansion::NonConst, 2>(variants_ch_ac);

  Test::Variants variants_ac = {{1, false},
                                {0, true},
                                {MAX_SINTERING_GRAINS + 1, false},
                                {MAX_SINTERING_GRAINS + 3, true}};
  Test::run_instantiation<Test::Expansion::NonConst, 0>(variants_ac);
}