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
      return nonscalar_ranges;
    }

    void
    wrap_range_to_vector(const unsigned int from,
                         const unsigned int to,
                         std::string        name)
    {
      nonscalar_ranges.emplace_back(
        from,
        to,
        name,
        DataComponentInterpretation::component_is_part_of_vector);
    }

  private:
    std::vector<
      std::tuple<unsigned int,
                 unsigned int,
                 std::string,
                 DataComponentInterpretation::DataComponentInterpretation>>
      nonscalar_ranges;
  };
} // namespace dealii