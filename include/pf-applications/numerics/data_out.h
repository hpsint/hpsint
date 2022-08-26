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
    get_nonscalar_data_ranges() const
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