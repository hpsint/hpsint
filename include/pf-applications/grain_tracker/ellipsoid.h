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

  /* This function gets the radii a, b, and c of an ellipsoid in 3D for the
   * given values values of the principal moments of inertia which are given by
   * I0 = 1/4 * M * b^2
   * I1 = 1/4 * M * a^2
   * where
   * M = density * pi * a * b
   * see
   * https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Classical_Mechanics_(Tatum)/02%3A_Moments_of_Inertia/2.20%3A_Ellipses_and_Ellipsoids
   */
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
  /* This function gets the radii a, b, and c of an ellipsoid in 3D for the
   * given values values of the principal moments of inertia which are given by
   * I0 = 1/5 * M * (b^2 + c^2)
   * I1 = 1/5 * M * (a^2 + c^2)
   * I2 = 1/5 * M * (a^2 + b^2)
   * where
   * M = 4/3 * density * pi * a * b * c
   * see https://scienceworld.wolfram.com/physics/MomentofInertiaEllipsoid.html
   */
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

  /* A class that handles the ellipsoid geometry.
   * An ellipsoid is written in a quadratic form as:
   * f(x) = 1/2 * x^T A x + b^T x + alpha = 0
   * where x is the the point vector in dim space.
   */
  template <int dim, typename Number = double>
  class Ellipsoid
  {
  public:
    // Build an ellipsoid from its center, radii and principal axes
    Ellipsoid(const Point<dim, Number> &                 center,
              const std::array<Number, dim> &            radii,
              const std::array<Point<dim, Number>, dim> &axes)
      : radii(radii)
      , r_min(*std::min_element(radii.cbegin(), radii.cend()))
      , r_max(*std::max_element(radii.cbegin(), radii.cend()))
      , axes(axes)
      , center(center)
    {
      Tensor<2, dim, Number> S;

      for (unsigned int d = 0; d < dim; ++d)
        S[d][d] = 1. / std::pow(radii[d], 2);

      // Build a rotation tensor from a global cartesian system to the local one
      const auto Q = rotation_tensor_from_axes<dim, Number>(axes);

      // Build the quadratic form
      A = Physics::Transformations::basis_transformation(S, Q);
      b = A * center;
      b *= -1;
      alpha = 0.5 * (A * center) * center - 0.5;

      norm = A.norm();
    }

    /* Build an ellipsoid from its center, principal moments, principal axes and
     * measure (i.e. volume). Effectively, we evaluated radii from the principal
     * moments and then delegate to the constructor above.
     */
    Ellipsoid(const Point<dim, Number> &                 principal_center,
              const std::array<Number, dim> &            principal_moments,
              const std::array<Point<dim, Number>, dim> &principal_axes,
              const Number                               measure)
      : Ellipsoid(principal_center,
                  get_radii_from_inertia<Number>(principal_moments, measure),
                  principal_axes)
    {}

    // Build an ellipsoid from its quadratic form
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

      mtrA.copy_from(A);
      b.unroll(vecB.data(), vecB.data() + dim);

      // Compute the eigenvalues and eigenvectors of A to get the radii and
      // principal axes
      SymmetricTensor<2, dim, Number> Am(A);
      Am *= 1. / (2 * alpha_in);
      auto evals_and_vectors = eigenvectors(Am);

      // Ensure the eigenvalues are in the ascending order so the radii later
      // are in the descending: this is for consistency only, the ellipsoid
      // algorithms are not dependent on this ordering in fact
      std::sort(evals_and_vectors.begin(),
                evals_and_vectors.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

      std::transform(evals_and_vectors.begin(),
                     evals_and_vectors.end(),
                     radii.begin(),
                     [](const auto &pair) {
                       return 1. / std::sqrt(pair.first);
                     });

      std::transform(evals_and_vectors.begin(),
                     evals_and_vectors.end(),
                     axes.begin(),
                     [](const auto &pair) {
                       return pair.second / pair.second.norm();
                     });

      r_min = *std::min_element(radii.cbegin(), radii.cend());
      r_max = *std::max_element(radii.cbegin(), radii.cend());

      // Invert A to compute the ellipsoid center
      mtrA_inv.invert(mtrA);

      Vector<Number> vecC(dim);
      mtrA_inv.vmult(vecC, vecB);

      std::transform(vecC.begin(),
                     vecC.end(),
                     center.begin_raw(),
                     [](Number v) { return -v; });
    }

    Ellipsoid() = default;

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &A;
      ar &b;
      ar &alpha;
      ar &norm;
      ar &radii;
      ar &r_min;
      ar &r_max;
      ar &center;
    }

    const Tensor<2, dim, Number> &
    get_A() const
    {
      return A;
    }

    const Tensor<1, dim, Number> &
    get_b() const
    {
      return b;
    }

    Number
    get_alpha() const
    {
      return alpha;
    }

    Number
    get_norm() const
    {
      return norm;
    }

    Number
    get_r_min() const
    {
      return r_min;
    }

    Number
    get_r_max() const
    {
      return r_max;
    }

    const std::array<Number, dim> &
    get_radii() const
    {
      return radii;
    }

    // Spectral radius (evaluated from the smallest radius)
    Number
    get_spectral() const
    {
      return std::pow(r_min, 2);
    }

    const Point<dim, Number> &
    get_center() const
    {
      return center;
    }

    const std::array<Point<dim, Number>, dim> &
    get_axes() const
    {
      return axes;
    }

  private:
    // Components of the quadratic form
    Tensor<2, dim, Number> A;
    Tensor<1, dim, Number> b;
    Number                 alpha;

    // Norm of matrix A
    Number norm;

    // Radii
    std::array<Number, dim> radii;

    // Cached min and max radii
    Number r_min;
    Number r_max;

    // Axes
    std::array<Point<dim, Number>, dim> axes;

    // Center of the ellipsoid
    Point<dim, Number> center;
  };

  namespace internal
  {
    // Solve a quadratic equation
    template <typename Number>
    std::pair<Number, Number>
    solve_quadratic(Number a, Number b, Number c)
    {
      const auto D = b * b - 4 * a * c;

      return {(-b - std::sqrt(D)) / (2 * a), (-b + std::sqrt(D)) / (2 * a)};
    }

    /* This function searches for an intersection of an ellipsoid E with a line
     * connecting points c1 and c2 and defined parametrically using variable t
     * as x(t) = c1 + t*(c2 - c1). Note that either c1 or c2 is located inside
     * the ellipsoid E. To solve this problem, one needs to substitute this
     * parametric definition into the ellipsoid quadratic form f(x) = 0. The
     * quadratic equation with unknown t is then obtained, and only one of its
     * solutions is in the admissible range [0, 1].
     */
    template <int dim, typename Number>
    std::pair<Number, bool>
    find_new_t(const Ellipsoid<dim, Number> &E,
               const Tensor<1, dim, Number> &c1,
               const Tensor<1, dim, Number> &c2)
    {
      const auto Ac1 = E.get_A() * c1;
      const auto Ac2 = E.get_A() * c2;

      const auto a = 0.5 * (Ac1 * c1 - 2 * Ac1 * c2 + Ac2 * c2);
      const auto b = Ac2 * c1 - Ac1 * c1 + E.get_b() * c2 - E.get_b() * c1;
      const auto c = 0.5 * Ac1 * c1 + E.get_b() * c1 + E.get_alpha();

      const auto sol = solve_quadratic(a, b, c);

      auto in_range = [](Number v) { return v >= 0. && v <= 1.; };

      // Then the ellipses already overlap or one contains the other
      if (!in_range(sol.first) && !in_range(sol.second))
        {
          return std::make_pair(0, true);
        }

      AssertThrow(!in_range(sol.first) || !in_range(sol.second),
                  ExcMessage("Both solutions " + std::to_string(sol.first) +
                             " and " + std::to_string(sol.second) +
                             " are in the admissible range"));

      return std::make_pair((in_range(sol.first)) ? sol.first : sol.second,
                            false);
    }

    /* One of the metric that can be ised for convergence, not used at the
     * moment since it has acos.
     */
    template <int dim, typename Number>
    Number
    theta(const Tensor<1, dim, Number> &v, const Tensor<1, dim, Number> &w)
    {
      return std::acos(v * w / (v.norm() + w.norm()));
    }

    /* This is one of the convergence metrics and it is estimates whether the 2
     * vectors v and w are collinear or not.
     */
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

  /* This function searches for the minimal distance between the 2 ellipsoids E1
   * and E1. It is based on the so-called "moving balls" algorithm proposed in
   * https://epubs.siam.org/doi/10.1137/S1052623401396510. It is sufficiently
   * fast and is very robust as shown in https://hal.science/hal-04507684.
   *
   * We use 2 different convergence criteria: either of them or both at the same
   * time can be used. The first one checks the relative magnitude of change of
   * the distance between the ellipsoids at the current iteration with respect
   * to the corresponding values from the previous. The second one checks the
   * collinearity of the 2 vectors, see works above for details.
   *
   * We return the distance itself, the number of iterations, and the flag that
   * says whether the iterative process has actually converged. If that flag is
   * false, the distance actually still remains meaningfull and, most
   * probably, is usable for our needs thanks to the robustness of this
   * "moving balls" algorithm.
   */
  template <int dim, typename Number>
  std::tuple<Number, unsigned int, bool>
  distance(const Ellipsoid<dim, Number> &E1,
           const Ellipsoid<dim, Number> &E2,
           const bool                    test_relative  = true,
           const bool                    test_collinear = false,
           const Number                  tol            = 1e-6,
           unsigned int                  max_iter       = 100)
  {
    AssertThrow(test_relative || test_collinear,
                ExcMessage(
                  "At least one convergence criteria should be chosen"));

    /* The initial reference points are simply the ellipsoid centers. */
    Tensor<1, dim, Number> c1 = E1.get_center();
    Tensor<1, dim, Number> c2 = E2.get_center();

    /* Convergence tolerance is reevaluated based on the maximum radii of the
     * ellipsoid curvatures.
     */
    const auto eps_d = std::sqrt(2 * tol / (E1.get_r_max() + E2.get_r_max()));

    const auto c2c1_init = c2 - c1;
    AssertThrow(c2c1_init.norm() > tol,
                ExcMessage("Ellipses centers coincide"));

    Number       dist          = c2c1_init.norm();
    unsigned int iter          = 0;
    bool         has_converged = false;

    for (; !has_converged && iter < max_iter; ++iter)
      {
        const Number dist_prev = dist;

        /* At each iteration we build a line connecting points c1 and c2
         * currently picked as reference ones within ellipsoids E1 and E1. This
         * line is defined in parametric way using variable t. We then search
         * for an intersection of the line with each of the ellipsoid using this
         * function. To this end
         */
        const auto [t1, overlap1] = internal::find_new_t(E1, c1, c2);
        if (overlap1)
          {
            dist          = 0;
            has_converged = true;
            break;
          }

        const auto [t2, overlap2] = internal::find_new_t(E2, c1, c2);
        if (overlap2 || t2 < t1)
          {
            dist          = 0;
            has_converged = true;
            break;
          }

        const auto c2c1 = c2 - c1;

        /* These are intersection of the ellipsoids with the line connecting
         * points c1 and c2.
         */
        const Point<dim, Number> x1_bar(c1 + t1 * c2c1);
        const Point<dim, Number> x2_bar(c1 + t2 * c2c1);

        /* This is interpreted as a distance between the ellipsoids. */
        dist = x1_bar.distance(x2_bar);

        /* Check the relative change of the distance between the iterations. */
        bool ok_relative = true;
        if (test_relative)
          {
            const auto rel_err = std::abs(dist - dist_prev) / dist_prev;
            ok_relative        = rel_err < tol;
          }

        /* Check if the vector connecting the obtained itersection points is
         * collinear with the vectors originating from the center of each of the
         * ellipsoids to the corresponding intersection point: these 3 should
         * all be collinear.
         */
        bool ok_collinear = true;
        if (test_collinear)
          {
            const auto crit1 = internal::crit(x2_bar - x1_bar,
                                              E1.get_A() * x1_bar + E1.get_b(),
                                              eps_d);
            const auto crit2 = internal::crit(x1_bar - x2_bar,
                                              E2.get_A() * x2_bar + E2.get_b(),
                                              eps_d);
            ok_collinear     = crit1 && crit2;
          }

        has_converged = ok_relative && ok_collinear;

        /* Pick new reference points. */
        c1 = x1_bar - 1. / E1.get_norm() * (E1.get_A() * x1_bar + E1.get_b());
        c2 = x2_bar - 1. / E2.get_norm() * (E2.get_A() * x2_bar + E2.get_b());
      }

    return {dist, iter, has_converged};
  }
} // namespace GrainTracker