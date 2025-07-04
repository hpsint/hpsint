// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <iostream>

namespace Sintering
{
  template <template <int dim, typename Number, typename VectorizedArrayType>
            typename NonLinearOperator,
            template <typename VectorizedArrayType>
            typename FreeEnergy>
  void
  runner(int           argc,
         char        **argv,
         std::ostream &out,
         std::ostream &out_statistics);
} // namespace Sintering