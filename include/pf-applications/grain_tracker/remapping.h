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

#pragma once

namespace GrainTracker
{
  struct Remapping
  {
    unsigned int grain_id;
    unsigned int from;
    unsigned int to;

    Remapping(const unsigned int grain_id,
              const unsigned int from,
              const unsigned int to)
      : grain_id(grain_id)
      , from(from)
      , to(to)
    {}

    bool
    operator==(const Remapping &rhs) const
    {
      return grain_id == rhs.grain_id && from == rhs.from && to == rhs.to;
    }

    bool
    operator!=(const Remapping &rhs) const
    {
      return grain_id != rhs.grain_id || from != rhs.from || to != rhs.to;
    }
  };
} // namespace GrainTracker
