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

#pragma once

#include <deal.II/base/subscriptor.h>

namespace GrainTracker
{
  using namespace dealii;

  class Mapper : public Subscriptor
  {
  public:
    virtual unsigned int
    get_particle_index(const unsigned int order_parameter,
                       const unsigned int cell_index) const = 0;

    virtual std::pair<unsigned int, unsigned int>
    get_grain_and_segment(const unsigned int order_parameter,
                          const unsigned int particle_id) const = 0;

    virtual unsigned int
    get_grain_segment_index(const unsigned int grain_id,
                            const unsigned int segment_id) const = 0;

    virtual unsigned int
    n_segments() const = 0;

    virtual bool
    empty() const = 0;
  };
} // namespace GrainTracker