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

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/physics/transformations.h>

#include <pf-applications/grain_tracker/motion.h>

#include <utility>

namespace GrainTracker
{
  using namespace dealii;

  template <typename Number>
  std::array<Number, 2>
  get_radii_from_inertia(const std::array<Number, 2> &principal_moments,
                         const Number                 measure)
  {
    const auto &I0 = principal_moments[0];
    const auto &I1 = principal_moments[1];

    std::array<Number, 2> radii;
    radii[0] = std::sqrt(4. / measure * I1);
    radii[1] = std::sqrt(4. / measure * I0);

    return radii;
  }

  template <typename Number>
  std::array<Number, 3>
  get_radii_from_inertia(const std::array<Number, 3> &principal_moments,
                         const Number                 measure)
  {
    const auto &I0 = principal_moments[0];
    const auto &I1 = principal_moments[1];
    const auto &I2 = principal_moments[2];

    std::array<Number, 3> radii;
    radii[0] = std::sqrt(2.5 / measure * (I1 + I2 - I0));
    radii[1] = std::sqrt(2.5 / measure * (I0 + I2 - I1));
    radii[2] = std::sqrt(2.5 / measure * (I0 + I1 - I2));

    return radii;
  }

  template <int dim, typename Number = double>
  struct Ellipsoid
  {
    Tensor<2, dim, Number> A;
    Tensor<1, dim, Number> b;
    Number                 alpha;

    // Sorted in the ascending order
    std::array<Number, dim> radii;

    Number             norm;
    Number             gamma;
    Point<dim, Number> center;

    Ellipsoid(const Point<dim, Number> &                 center,
              const std::array<Number, dim> &            radii,
              const std::array<Point<dim, Number>, dim> &axes)
      : radii(radii)
      , center(center)
    {
      Tensor<2, dim, Number> S;

      for (unsigned int d = 0; d < dim; ++d)
        S[d][d] = 1. / std::pow(radii[d], 2);

      const auto Q = rotation_tensor_from_axes<dim, Number>(axes);

      A = Physics::Transformations::basis_transformation(S, Q);
      b = A * center;
      b *= -1;
      alpha = 0.5 * (A * center) * center - 0.5;

      norm = A.norm();

      gamma = std::pow(min_radius(), 2);
    }

    Ellipsoid(const Point<dim, Number> &                 principal_center,
              const std::array<Number, dim> &            principal_moments,
              const std::array<Point<dim, Number>, dim> &principal_axes,
              const Number                               measure)
      : Ellipsoid(principal_center,
                  get_radii_from_inertia<dim, Number>(principal_moments,
                                                      measure),
                  principal_axes)
    {}

    Ellipsoid(const SymmetricTensor<2, dim, Number> &A_in,
              const Tensor<1, dim, Number> &         b_in,
              const Number                           alpha_in)
      : A(A_in)
      , b(b_in)
      , alpha(alpha_in)
      , norm(A.norm())
    {
      FullMatrix<Number> mtrA(dim, dim);
      FullMatrix<Number> mtrA_inv(dim, dim);
      Vector<Number>     vecB(dim);

      mtrA.fill(A.begin_raw());
      std::copy_n(b.begin_raw(), dim, vecB.data());

      SymmetricTensor<2, dim, Number> Am(A);
      Am *= 1. / (2 * alpha_in);
      const auto evals = eigenvalues(Am);

      std::transform(evals.begin(), evals.end(), radii.begin(), [](Number v) {
        return 1. / std::sqrt(v);
      });

      gamma = std::pow(min_radius(), 2);

      mtrA_inv.invert(mtrA);

      Vector<Number> vecC(dim);
      mtrA_inv.vmult(vecC, vecB);

      std::transform(vecC.begin(),
                     vecC.end(),
                     center.begin_raw(),
                     [](Number v) { return -v; });
    }

    Ellipsoid() = default;

    Number
    min_radius() const
    {
      return *std::min_element(radii.cbegin(), radii.cend());
    }

    Number
    max_radius() const
    {
      return *std::max_element(radii.cbegin(), radii.cend());
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &A;
      ar &b;
      ar &alpha;
      ar &radii;
      ar &norm;
      ar &gamma;
      ar &center;
    }
  };

} // namespace GrainTracker