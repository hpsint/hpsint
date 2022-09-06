#pragma once

#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename VectorizedArrayType>
  class AdvectionMechanism
  {
  public:
    template <bool level_dof_access>
    void
    reinit(const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>>
             &cell) const
    {
      (void)cell;
    }

    void
    reinit(const unsigned int cell_index) const
    {
      (void)cell_index;
    }

    const Tensor<1, dim, VectorizedArrayType> &
    get_velocity(const Point<dim, VectorizedArrayType> p) const
    {
      (void)p;
      return current_velocity;
    }

    const Tensor<1, dim, VectorizedArrayType> &
    get_velocity_derivative(const Point<dim, VectorizedArrayType> p) const
    {
      (void)p;
      return current_velocity_derivative;
    }

    unsigned int
    get_order_parameter_id() const
    {
      return current_order_parameter_id;
    }

  private:
    mutable Tensor<1, dim, VectorizedArrayType> current_velocity;
    mutable Tensor<1, dim, VectorizedArrayType> current_velocity_derivative;
    mutable unsigned int                        current_order_parameter_id =
      numbers::invalid_unsigned_int;
  };
} // namespace Sintering