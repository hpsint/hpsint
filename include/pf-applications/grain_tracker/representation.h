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

#include <pf-applications/base/debug.h>

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

#include <pf-applications/grain_tracker/ellipsoid.h>

namespace GrainTracker
{
  using namespace dealii;

  // This base erases type
  struct Representation
  {
    virtual double
    distance(const Representation &other) const = 0;

    virtual void
    print(std::ostream &stream) const = 0;

    virtual std::unique_ptr<Representation>
    clone() const = 0;

    virtual ~Representation()
    {}

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      (void)ar;
    }
  };

  // The CRTP base that handles static polymorphism
  template <typename T>
  struct RepresentationWrapper : Representation
  {
    /* Though this CRTP behavior does not introduce any runtime overhead, it is
     * not absolutely safe, since other object can be of different type. For
     * instance, one may think of measuring distance between a sphere and an
     * ellipsoid. This is currently not permitted and will lead to incorrect
     * static_cast behavior. */
    double
    distance(const Representation &other) const override
    {
      // This expensive check is used only in debug builds
      Assert(dynamic_cast<const T *>(&other), ExcNotImplemented());

      return static_cast<const T *>(this)->distance_impl(
        static_cast<const T &>(other));
    }
  };

  template <int dim>
  struct RepresentationSpherical
    : public RepresentationWrapper<RepresentationSpherical<dim>>
  {
    RepresentationSpherical(const Point<dim> &center_in, const double radius_in)
      : center(center_in)
      , radius(radius_in)
    {}

    RepresentationSpherical() = default;

    double
    distance_impl(const RepresentationSpherical<dim> &other) const
    {
      const double distance_centers = center.distance(other.center);
      const double sum_radii        = radius + other.radius;

      const double current_distance = distance_centers - sum_radii;

      return current_distance;
    }

    void
    print(std::ostream &stream) const override
    {
      stream << "center = " << center << " | radius = " << radius;
    }

    std::unique_ptr<Representation>
    clone() const override
    {
      return std::make_unique<RepresentationSpherical>(center, radius);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &BOOST_SERIALIZATION_BASE_OBJECT_NVP(Representation);
      ar &center;
      ar &radius;
    }

    Point<dim> center;
    double     radius;
  };

  template <int dim>
  struct RepresentationElliptical
    : public RepresentationWrapper<RepresentationElliptical<dim>>
  {
    RepresentationElliptical(const Point<dim> &center,
                             const double      measure,
                             const double *    data)
    {
      double inertia_data[num_inertias<dim>];
      std::copy_n(data, num_inertias<dim>, inertia_data);

      SymmetricTensor<2, dim> inertia(inertia_data);

      initialize(center, measure, inertia);
    }

    RepresentationElliptical(const Point<dim> &             center,
                             const double                   measure,
                             const SymmetricTensor<2, dim> &inertia)
    {
      initialize(center, measure, inertia);
    }

    RepresentationElliptical(const Point<dim> &                 center,
                             const std::array<double, dim> &    radii,
                             const std::array<Point<dim>, dim> &axes)
      : ellipsoid(center, radii, axes)
    {}

    RepresentationElliptical(const Ellipsoid<dim> &ellipsoid)
      : ellipsoid(ellipsoid)
    {}

    RepresentationElliptical() = default;

    double
    distance_impl(const RepresentationElliptical<dim> &other) const
    {
      const auto res = distance(this->ellipsoid, other.ellipsoid);

      return std::get<0>(res);
    }

    void
    print(std::ostream &stream) const override
    {
      stream << "center = " << ellipsoid.get_center() << " | radii = ";
      stream << debug::to_string(ellipsoid.get_radii().begin(),
                                 ellipsoid.get_radii().end(),
                                 " ");
    }

    virtual std::unique_ptr<Representation>
    clone() const override
    {
      return std::make_unique<RepresentationElliptical>(ellipsoid);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &BOOST_SERIALIZATION_BASE_OBJECT_NVP(Representation);
      ar &ellipsoid;
    }

  private:
    void
    initialize(const Point<dim> &             center,
               const double                   measure,
               const SymmetricTensor<2, dim> &inertia)
    {
      auto evals_and_vectors = eigenvectors(inertia);

      // Check that the eigenvalues are in the descending order
      std::sort(evals_and_vectors.begin(),
                evals_and_vectors.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

      std::array<double, dim>     principal_moments;
      std::array<Point<dim>, dim> principal_axes;

      for (unsigned int d = 0; d < dim; ++d)
        {
          auto p = Point<dim>(evals_and_vectors[d].second);
          p /= p.norm();

          principal_axes[d] = p;

          principal_moments[d] = evals_and_vectors[d].first;
        }

      ellipsoid = std::move(
        Ellipsoid<dim>(center, principal_moments, principal_axes, measure));
    }

  public:
    Ellipsoid<dim> ellipsoid;
  };
} // namespace GrainTracker
