#pragma once

#include <deal.II/base/tensor.h>

namespace Structural
{
  using namespace dealii;

  template <int dim>
  constexpr unsigned int voigt_size = dim *((dim + 1) / 2.);

  namespace internal
  {
    template <int dim>
    struct VoigtIndices
    {};

    template <>
    struct VoigtIndices<2>
    {
      static constexpr std::array<const char *, voigt_size<2>> indices = {
        {"eps_xx", "eps_yy", "tau_xy"}};
    };

    template <>
    struct VoigtIndices<3>
    {
      static constexpr std::array<const char *, voigt_size<3>> indices = {
        {"eps_xx", "eps_yy", "eps_zz", "tau_xy", "tau_yz", "tau_xz"}};
    };
  } // namespace internal

  template <int dim>
  inline constexpr auto voigt_indices = internal::VoigtIndices<dim>::indices;

  template <int dim, typename Number>
  Tensor<1, voigt_size<dim>, Number>
  apply_l(Tensor<2, dim, Number> gradient_in)
  {
    if constexpr (dim == 2)
      {
        Tensor<1, voigt_size<dim>, Number> vector_in;
        vector_in[0] = gradient_in[0][0];
        vector_in[1] = gradient_in[1][1];
        vector_in[2] = gradient_in[1][0] + gradient_in[0][1];
        return vector_in;
      }
    else if constexpr (dim == 3)
      {
        Tensor<1, voigt_size<dim>, Number> vector_in;
        vector_in[0] = gradient_in[0][0];
        vector_in[1] = gradient_in[1][1];
        vector_in[2] = gradient_in[2][2];
        vector_in[3] = gradient_in[0][1] + gradient_in[1][0];
        vector_in[4] = gradient_in[1][2] + gradient_in[2][1];
        vector_in[5] = gradient_in[0][2] + gradient_in[2][0];
        return vector_in;
      }
    else
      {
        AssertThrow(dim == 2 || dim == 3, ExcNotImplemented());
      }
  }

  template <int dim, typename Number>
  Tensor<2, dim, Number>
  apply_l_transposed(Tensor<1, voigt_size<dim>, Number> vector_out)
  {
    if constexpr (dim == 2)
      {
        Tensor<2, dim, Number> gradient_out;
        gradient_out[0][0] = vector_out[0];
        gradient_out[1][1] = vector_out[1];

        gradient_out[0][1] = vector_out[2];
        gradient_out[1][0] = vector_out[2];
        return gradient_out;
      }
    else if constexpr (dim == 3)
      {
        Tensor<2, dim, Number> gradient_out;
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
    else
      {
        AssertThrow(dim == 2 || dim == 3, ExcNotImplemented());
      }
  }

  template <int dim, typename Number>
  Tensor<2, dim, Number>
  add_identity(Tensor<2, dim, Number> gradient)
  {
    for (unsigned int i = 0; i < dim; i++)
      gradient[i][i] = gradient[i][i] + 1.0;
    return gradient;
  }

  template <int dim, typename Number>
  Tensor<2, dim, Number>
  sub_identity(Tensor<2, dim, Number> gradient)
  {
    for (unsigned int i = 0; i < dim; i++)
      gradient[i][i] = gradient[i][i] - 1.0;
    return gradient;
  }
} // namespace Structural