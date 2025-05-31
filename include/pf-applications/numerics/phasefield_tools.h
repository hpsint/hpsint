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

#include <deal.II/fe/component_mask.h>

namespace hpsint
{
  template <typename BlockVectorType>
  void
  limit_vector_values(BlockVectorType     &vec,
                      const ComponentMask &mask = ComponentMask(),
                      const typename BlockVectorType::value_type val_min = 0,
                      const typename BlockVectorType::value_type val_max = 1)
  {
    AssertThrow(mask.represents_the_all_selected_mask() ||
                  vec.n_blocks() == mask.size(),
                ExcMessage("Vector size (" + std::to_string(vec.n_blocks()) +
                           ") does not fit the mask size (" +
                           std::to_string(mask.size()) + ")."));

    for (unsigned int b = 0; b < vec.n_blocks(); ++b)
      if (mask[b])
        for (auto &val : vec.block(b))
          {
            if (val < val_min)
              val = val_min;
            else if (val > val_max)
              val = val_max;
          }
  }
} // namespace hpsint