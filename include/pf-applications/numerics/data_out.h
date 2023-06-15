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

#include <deal.II/numerics/data_out.h>

namespace dealii
{
  template <int dim>
  class DataOutWithRanges : public DataOut<dim>
  {
  public:
    std::vector<
      std::tuple<unsigned int,
                 unsigned int,
                 std::string,
                 DataComponentInterpretation::DataComponentInterpretation>>
    get_nonscalar_data_ranges() const override
    {
      auto ranges = DataOut<dim>::get_nonscalar_data_ranges();

      auto names = this->get_dataset_names();

      for (unsigned int i = 0; i < names.size();)
        {
          unsigned int n_components = 1;
          for (unsigned int j = i + 1; j < names.size(); ++j)
            if (names[j] == names[i])
              n_components++;
            else
              break;

          if (n_components == dim)
            ranges.emplace_back(
              i,
              i + n_components - 1,
              names[i],
              DataComponentInterpretation::component_is_part_of_vector);
          else if (n_components == dim * dim)
            ranges.emplace_back(
              i,
              i + n_components - 1,
              names[i],
              DataComponentInterpretation::component_is_part_of_tensor);

          i += n_components;
        }

      return ranges;
    }
  };
} // namespace dealii