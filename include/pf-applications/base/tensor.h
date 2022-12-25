#pragma once

#include <deal.II/base/tensor.h>

namespace Sintering
{
  using namespace dealii;

  namespace internal
  {
    template <int dim, typename Number>
    struct Moment
    {};

    template <typename Number>
    struct Moment<2, Number>
    {
      typedef Number                type;
      static constexpr unsigned int size = 1;
    };

    template <typename Number>
    struct Moment<3, Number>
    {
      typedef Tensor<1, 3, Number>  type;
      static constexpr unsigned int size = 3;
    };

  } // namespace internal

  template <int dim, typename Number>
  using moment_t = typename internal::Moment<dim, Number>::type;

  template <int dim, typename Number>
  inline constexpr unsigned int moment_s = internal::Moment<dim, Number>::size;

  template <int dim, typename Number>
  moment_t<dim, Number>
  cross_product(const Tensor<1, dim, Number> &a,
                const Tensor<1, dim, Number> &b);

  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline moment_t<2, Number>
  cross_product(const Tensor<1, 2, Number> &a, const Tensor<1, 2, Number> &b)
  {
    return a[1] * b[0] - a[0] * b[1];
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline auto
  create_moment_from_buffer(const Number *buffer)
  {
    if constexpr (dim == 3)
      return Tensor<1, 3, Number>(make_array_view(buffer, buffer + dim));
    else
      return *buffer;
  }

  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<1, 3, Number>
  cross_product(const Tensor<1, 3, Number> &a, const Tensor<1, 3, Number> &b)
  {
    return cross_product_3d(a, b);
  }

  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<1, 2, Number>
  cross_product(const Number &a, const Tensor<1, 2, Number> &b)
  {
    Tensor<1, 2, Number> c;

    c[0] = -b[1];
    c[1] = b[0];
    c *= a;

    return c;
  }

  // Compute skew tensor of a vector
  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<2, 3, Number>
  skew(const Tensor<1, 3, Number> &a)
  {
    Tensor<2, 3, Number> A;
    A[0][1] = -a[2];
    A[0][2] = a[1];
    A[1][0] = a[2];
    A[1][2] = -a[0];
    A[2][0] = -a[1];
    A[2][1] = a[0];

    return A;
  }

  template <typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<1, 2, Number>
  skew(const Tensor<1, 2, Number> &a)
  {
    Tensor<1, 2, Number> A;
    A[0] = a[1];
    A[1] = -a[0];

    return A;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<2, dim, Number>
  diagonal_matrix(const Number &fac = 1.)
  {
    Tensor<2, dim, Number> I;

    for (unsigned int d = 0; d < dim; d++)
      I[d][d] = fac;

    return I;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, Number>
  unit_vector(const Tensor<1, dim, Number> &vec)
  {
    Number nrm = vec.norm();
    Number filter;

    Number zeros(0.0);
    Number ones(1.0);
    Number zero_tol(1e-4);

    Tensor<1, dim, Number> n = vec;

    filter = compare_and_apply_mask<SIMDComparison::greater_than>(nrm,
                                                                  zero_tol,
                                                                  ones,
                                                                  zeros);
    nrm    = compare_and_apply_mask<SIMDComparison::less_than>(nrm,
                                                            zero_tol,
                                                            ones,
                                                            nrm);

    n /= nrm;
    n *= filter;

    return n;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<2, dim, Number>
  projector_matrix(const Tensor<1, dim, Number> vec, const Number &fac = 1.)
  {
    auto tensor = diagonal_matrix<dim, Number>(1.) - outer_product(vec, vec);
    tensor *= fac;

    return tensor;
  }

} // namespace Sintering
