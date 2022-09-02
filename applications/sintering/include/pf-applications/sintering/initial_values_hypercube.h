#pragma once

#include "initial_values_array.h"

namespace Sintering
{
  template <int dim>
  class InitialValuesHypercube : public InitialValuesArray<dim>
  {
  public:
    InitialValuesHypercube(const double                         r0,
                           const double                         interface_width,
                           const std::array<unsigned int, dim> &n_grains,
                           const bool minimize_order_parameters,
                           const bool is_accumulative)
      : InitialValuesArray<dim>(r0, interface_width, is_accumulative)
    {
      unsigned int counter = 0;

      if (minimize_order_parameters)
        {
          this->order_parameter_to_grains[0];
          if (std::any_of(n_grains.cbegin(),
                          n_grains.cend(),
                          [](const auto &val) { return val > 1; }))
            {
              this->order_parameter_to_grains[1];
            }
        }

      if (dim == 2)
        {
          for (unsigned int i = 0; i < n_grains[0]; ++i)
            for (unsigned int j = 0; j < n_grains[1]; ++j)
              {
                this->centers.emplace_back(2 * r0 * i, 2 * r0 * j);
                assign_order_parameter(i + j,
                                       counter++,
                                       minimize_order_parameters);
              }
        }
      else if (dim == 3)
        {
          for (unsigned int i = 0; i < n_grains[0]; ++i)
            for (unsigned int j = 0; j < n_grains[1]; ++j)
              for (unsigned int k = 0; k < n_grains[2]; ++k)
                {
                  this->centers.emplace_back(2 * r0 * i,
                                             2 * r0 * j,
                                             2 * r0 * k);
                  assign_order_parameter(i + j + k,
                                         counter++,
                                         minimize_order_parameters);
                }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

  private:
    void
    assign_order_parameter(const unsigned int order,
                           const unsigned int counter,
                           const bool         minimize_order_parameters)
    {
      if (minimize_order_parameters)
        this->order_parameter_to_grains[order % 2].push_back(counter);
      else
        this->order_parameter_to_grains[counter] = {counter};
    }
  };
} // namespace Sintering