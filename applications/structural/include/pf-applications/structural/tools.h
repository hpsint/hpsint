#pragma once

#include <deal.II/base/tensor.h>

namespace Structural
{
  using namespace dealii;

  template <int dim>
  constexpr unsigned int voigt_size = dim *((dim + 1) / 2.);

  template <int dim, typename VectorizedArrayType>
  Tensor<1, voigt_size<dim>, VectorizedArrayType>
  apply_l(Tensor<2, dim, VectorizedArrayType> gradient_in)
  {
    if constexpr (dim == 2)
      {
        Tensor<1, voigt_size<dim>, VectorizedArrayType> vector_in;
        vector_in[0] = gradient_in[0][0];
        vector_in[1] = gradient_in[1][1];
        vector_in[2] = gradient_in[1][0] + gradient_in[0][1];
        return vector_in;
      }
    else // dim==3
      {
        Tensor<1, voigt_size<dim>, VectorizedArrayType> vector_in;
        vector_in[0] = gradient_in[0][0];
        vector_in[1] = gradient_in[1][1];
        vector_in[2] = gradient_in[2][2];
        vector_in[3] = gradient_in[0][1] + gradient_in[1][0];
        vector_in[4] = gradient_in[1][2] + gradient_in[2][1];
        vector_in[5] = gradient_in[0][2] + gradient_in[2][0];
        return vector_in;
      }
  }

  template <int dim, typename VectorizedArrayType>
  Tensor<2, dim, VectorizedArrayType>
  apply_l_transposed(Tensor<1, voigt_size<dim>, VectorizedArrayType> vector_out)
  {
    if constexpr (dim == 2)
      {
        Tensor<2, dim, VectorizedArrayType> gradient_out;
        gradient_out[0][0] = vector_out[0];
        gradient_out[1][1] = vector_out[1];

        gradient_out[0][1] = vector_out[2];
        gradient_out[1][0] = vector_out[2];
        return gradient_out;
      }
    else // dim==3
      {
        Tensor<2, dim, VectorizedArrayType> gradient_out;
        gradient_out[0][0] = vector_out[0];
        gradient_out[1][1] = vector_out[1];
        gradient_out[2][2] = vector_out[2];

        gradient_out[0][1] = vector_out[3];
        gradient_out[1][0] = vector_out[3];

        gradient_out[1][2] = vector_out[4];
        gradient_out[2][1] = vector_out[4];

        gradient_out[0][2] = vector_out[5];
        gradient_out[2][0] = vector_out[5];
        return gradient_out;
      }
  }
} // namespace Structural