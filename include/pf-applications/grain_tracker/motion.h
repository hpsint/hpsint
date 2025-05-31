// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#include <deal.II/base/point.h>

#include <array>

namespace GrainTracker
{
  using namespace dealii;

  template <int dim>
  constexpr unsigned int num_inertias = dim * ((dim + 1) / 2.);

  /* Build a rotation tensor between the two given coordinate systems each given
   * by unitary vectors e_i and e_i0 as Q = e_i e_i0^T. */
  template <int dim, typename Number>
  Tensor<2, dim, Number>
  rotation_tensor_from_axes(const std::array<Point<dim, Number>, dim> &axes,
                            const std::array<Point<dim, Number>, dim> &axes0)
  {
    Tensor<2, dim, Number> rotation_tensor;

    for (unsigned int d = 0; d < dim; ++d)
      rotation_tensor += outer_product(axes0[d], axes[d]);

    return rotation_tensor;
  }

  /* Similar to the previous function but implies that one of the coordinate
   * systems is actually the global one. */
  template <int dim, typename Number>
  Tensor<2, dim, Number>
  rotation_tensor_from_axes(const std::array<Point<dim, Number>, dim> &axes,
                            bool transpose = false)
  {
    std::array<Point<dim, Number>, dim> axes0;
    for (unsigned int d = 0; d < dim; ++d)
      axes0[d][d] = 1.;

    return transpose ? rotation_tensor_from_axes<dim, Number>(axes, axes0) :
                       rotation_tensor_from_axes<dim, Number>(axes0, axes);
  }

  /* Evaluate inertia moments in 2D */
  template <typename Number>
  void
  evaluate_inertia_properties(const Point<2, Number> &r,
                              const Number            measure,
                              Number                 *data)
  {
    const auto x = r[0];
    const auto y = r[1];

    data[0] += y * y * measure;
    data[1] += x * x * measure;
    data[2] += -x * y * measure;
  }

  /* Evaluate inertia moments in 3D */
  template <typename Number>
  void
  evaluate_inertia_properties(const Point<3, Number> &r,
                              const Number            measure,
                              Number                 *data)
  {
    const auto x = r[0];
    const auto y = r[1];
    const auto z = r[2];

    data[0] += (y * y + z * z) * measure;
    data[1] += (x * x + z * z) * measure;
    data[2] += (x * x + y * y) * measure;
    data[3] += -x * y * measure;
    data[4] += -x * z * measure;
    data[5] += -y * z * measure;
  }

  /* Compute a rotation tensor from a given rotation axis and a scalar angle in
   * 3D using quaternions. */
  template <typename Number>
  Tensor<2, 3, Number>
  rotation_via_quaternions(const Tensor<1, 3, Number> &axis, const Number angle)
  {
    const Number q0 = std::cos(angle / 2.);
    const Number q1 = std::sin(angle / 2.) * axis[0];
    const Number q2 = std::sin(angle / 2.) * axis[1];
    const Number q3 = std::sin(angle / 2.) * axis[2];

    Tensor<2, 3, Number> U;
    U[0][0] = 1 - 2. * (q2 * q2 + q3 * q3);
    U[0][1] = 2. * (q1 * q2 - q0 * q3);
    U[0][2] = 2. * (q0 * q2 + q1 * q3);
    U[1][0] = 2. * (q1 * q2 + q0 * q3);
    U[1][1] = 1 - 2. * (q1 * q1 + q3 * q3);
    U[1][2] = 2. * (q2 * q3 - q0 * q1);
    U[2][0] = 2. * (q1 * q3 - q0 * q2);
    U[2][1] = 2. * (q0 * q1 + q2 * q3);
    U[2][2] = 1 - 2. * (q1 * q1 + q2 * q2);

    return U;
  }
} // namespace GrainTracker