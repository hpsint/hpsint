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
                  get_radii_from_inertia<Number>(principal_moments, measure),
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

  namespace internal
  {
    template <typename Number>
    std::pair<Number, Number>
    solve_quadratic(Number a, Number b, Number c)
    {
      const auto D = b * b - 4 * a * c;

      return {(-b - std::sqrt(D)) / (2 * a), (-b + std::sqrt(D)) / (2 * a)};
    }

    template <int dim, typename Number>
    Number
    find_new_t(const Ellipsoid<dim, Number> &E,
               const Tensor<1, dim, Number> &c1,
               const Tensor<1, dim, Number> &c2)
    {
      const auto c2c1 = c2 - c1;
      // const auto a3 = 0.5 * (E.A * c2c1) * c2c1;
      // const auto b3 = (E.As * c2c1) * c1 + E.b * c2c1;
      // const auto c3 = 0.5 * (E.A * c1) * c1 + E.b * c1 + E.alpha;

      const auto Ac1 = E.A * c1;
      const auto Ac2 = E.A * c2;

      const auto a = 0.5 * (Ac1 * c1 - 2 * Ac1 * c2 + Ac2 * c2);
      const auto b = Ac2 * c1 - Ac1 * c1 + E.b * c2 - E.b * c1;
      const auto c = 0.5 * Ac1 * c1 + E.b * c1 + E.alpha;

      Number a0, b0, c0;
      if constexpr (dim == 2)
        {
          a0 = 0.5 *
               (E.A[0][0] * c2c1[0] * c2c1[0] + E.A[1][1] * c2c1[1] * c2c1[1] +
                2 * E.A[0][1] * c2c1[0] * c2c1[1]);
          b0 = (E.A[0][0] * c2c1[0]) * c1[0] + (E.A[1][1] * c2c1[1]) * c1[1] +
               E.A[0][1] * (c1[0] * c2c1[1] - c1[1] * c2c1[0]) + E.b * c2c1;
          c0 = 0.5 * (E.A[0][0] * c1[0] * c1[0] + E.A[1][1] * c1[1] * c1[1]) +
               E.A[0][1] * c1[0] * c1[1] + E.b * c1 + E.alpha;
        }
      else
        {
          a0 =
            0.5 *
            (E.A[0][0] * c2c1[0] * c2c1[0] + E.A[1][1] * c2c1[1] * c2c1[1] +
             E.A[2][2] * c2c1[2] * c2c1[2] + 2 * E.A[0][1] * c2c1[0] * c2c1[1] +
             2 * E.A[0][2] * c2c1[0] * c2c1[2] +
             2 * E.A[1][2] * c2c1[1] * c2c1[2]);
          b0 = (E.A[0][0] * c2c1[0]) * c1[0] + (E.A[1][1] * c2c1[1]) * c1[1] +
               (E.A[2][2] * c2c1[2]) * c1[2] +
               E.A[0][1] * (c1[0] * c2c1[1] - c1[1] * c2c1[0]) +
               E.A[0][2] * (c1[0] * c2c1[2] - c1[2] * c2c1[0]) +
               E.A[1][2] * (c1[1] * c2c1[2] - c1[2] * c2c1[1]) + E.b * c2c1;
          c0 = 0.5 * (E.A[0][0] * c1[0] * c1[0] + E.A[1][1] * c1[1] * c1[1] +
                      E.A[2][2] * c1[2] * c1[2]) +
               E.A[0][1] * c1[0] * c1[1] + E.A[0][2] * c1[0] * c1[2] +
               E.A[1][2] * c1[1] * c1[2] + E.b * c1 + E.alpha;
        }

      const auto sol = solve_quadratic(a, b, c);

      auto in_range = [](Number v) { return v >= 0. && v <= 1.; };

      AssertThrow(!(in_range(sol.first) && in_range(sol.second)),
                  ExcMessage("Both solutions " + std::to_string(sol.first) +
                             " and " + std::to_string(sol.second) +
                             " are in the admissible range"));
      AssertThrow(!(!in_range(sol.first) && !in_range(sol.second)),
                  ExcMessage("None of the solutions " +
                             std::to_string(sol.first) + " and " +
                             std::to_string(sol.second) +
                             " is in the admissible range"));

      return (in_range(sol.first)) ? sol.first : sol.second;
    }

    template <int dim, typename Number>
    Number
    theta(const Tensor<1, dim, Number> &v, const Tensor<1, dim, Number> &w)
    {
      return std::acos(v * w / (v.norm() + w.norm()));
    }

    template <int dim, typename Number>
    bool
    crit(const Tensor<1, dim, Number> &v,
         const Tensor<1, dim, Number> &w,
         const Number                  eps)
    {
      return std::pow(v * w, 2) > std::pow(1 - eps * eps / 2., 2) *
                                    v.norm_square() * w.norm_square();
    }
  } // namespace internal

  template <int dim, typename Number>
  std::pair<Number, unsigned int>
  distance(const Ellipsoid<dim, Number> &E1,
           const Ellipsoid<dim, Number> &E2,
           const Number                  tol      = 1e-10,
           unsigned int                  max_iter = 100)
  {
    Tensor<1, dim, Number> c1 = E1.center;
    Tensor<1, dim, Number> c2 = E2.center;

    const auto eps_d = std::sqrt(2 * tol / (E1.max_radius() + E2.max_radius()));

    const auto c2c1_init = c2 - c1;
    AssertThrow(c2c1_init.norm() > tol,
                ExcMessage("Ellipses centers coincide"));

    Number       dist = std::numeric_limits<Number>::max();
    unsigned int iter = 0;

    for (; iter < max_iter; ++iter)
      {
        const auto c2c1 = c2 - c1;
        const auto t1   = internal::find_new_t(E1, c1, c2);
        const auto t2   = internal::find_new_t(E2, c1, c2);

        if (t2 < t1)
          {
            dist = 0;
            break;
          }

        const Point<dim, Number> x1_bar(c1 + t1 * c2c1);
        const Point<dim, Number> x2_bar(c1 + t2 * c2c1);

        const auto crit1 =
          internal::crit(x2_bar - x1_bar, E1.A * x1_bar + E1.b, eps_d);
        const auto crit2 =
          internal::crit(x1_bar - x2_bar, E2.A * x2_bar + E2.b, eps_d);

        dist = x1_bar.distance(x2_bar);

        if (crit1 && crit2)
          {
            dist = x1_bar.distance(x2_bar);
            break;
          }

        c1 = x1_bar - 1. / E1.norm * (E1.A * x1_bar + E1.b);
        c2 = x2_bar - 1. / E2.norm * (E2.A * x2_bar + E2.b);
      }

    return {dist, iter};
  }
} // namespace GrainTracker