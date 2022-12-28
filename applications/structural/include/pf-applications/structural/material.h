#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <pf-applications/structural/tools.h>

namespace Structural
{
  using namespace dealii;

  enum class TWO_DIM_TYPE
  {
    PLAIN_STRESS,
    PLAIN_STRAIN,
    NONE
  };

  template <int dim, typename VectorizedArrayType>
  class Material
  {
  public:
    virtual Tensor<2, dim, VectorizedArrayType>
    get_S(const Tensor<2, dim, VectorizedArrayType> &deformation) const = 0;
  };

} // namespace Structural
