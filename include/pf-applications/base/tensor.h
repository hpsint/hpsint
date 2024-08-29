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

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace hpsint
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
  unit_vector(const Tensor<1, dim, Number> &vec, const Number zero_tol = 1e-10)
  {
    const Number zeros = 0.0;
    const Number ones  = 1.0;
    const Number nrm   = vec.norm();

    const auto filter = compare_and_apply_mask<SIMDComparison::greater_than>(
      nrm, zero_tol, ones, zeros);
    const auto filtered_norm =
      compare_and_apply_mask<SIMDComparison::less_than>(nrm,
                                                        zero_tol,
                                                        ones,
                                                        nrm);

    return (vec / filtered_norm) * filter;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Number
  unit_vector_filter_2(const Tensor<1, dim, Number> &vec,
                       const Number                  zero_tol = 1e-20)
  {
    const Number zeros = 0.0;
    const Number ones  = 1.0;
    const Number nrm   = vec.norm_square();

    const auto filter = compare_and_apply_mask<SIMDComparison::greater_than>(
      nrm, zero_tol, ones, zeros);
    const auto filtered_norm =
      compare_and_apply_mask<SIMDComparison::less_than>(nrm,
                                                        zero_tol,
                                                        ones,
                                                        nrm);

    return (filter / filtered_norm);
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline Tensor<2, dim, Number>
  projector_matrix(const Tensor<1, dim, Number> vec,
                   const Number &               fac  = 1.,
                   const Number &               diag = 1.)
  {
    auto tensor = diagonal_matrix<dim, Number>(diag) - outer_product(vec, vec);
    tensor *= fac;

    return tensor;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::array<Point<dim, Number>, dim>
  tensor_to_point_array(const Tensor<2, dim, Number> &tens)
  {
    std::array<Point<dim, Number>, dim> arr;

    std::copy_n(tens.begin_raw(), dim * dim, arr.begin()->begin_raw());

    return arr;
  }

  template <int dim, typename Number>
  DEAL_II_ALWAYS_INLINE inline std::array<Point<dim, Number>, dim>
  tensor_to_point_array(Tensor<2, dim, Number> &&tens)
  {
    std::array<Point<dim, Number>, dim> arr;

    std::copy_n(std::make_move_iterator(tens.begin_raw()),
                dim * dim,
                arr.begin());

    return arr;
  }
} // namespace hpsint
