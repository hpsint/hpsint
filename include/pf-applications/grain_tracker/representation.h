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

#include <boost/serialization/export.hpp>
#include <boost/serialization/serialization.hpp>

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
    double
    distance(const Representation &other) const override
    {
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

    virtual std::unique_ptr<Representation>
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

} // namespace GrainTracker

// Explicitly export intantiations to make polymorphic serialization work
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<2>);
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<3>);
