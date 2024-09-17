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

#include <deal.II/base/point.h>

#include <pf-applications/base/output.h>

#include <boost/serialization/unique_ptr.hpp>

#include <pf-applications/grain_tracker/representation.h>

#include <memory>

namespace GrainTracker
{
  using namespace dealii;

  /* Segment is a part of a grain created from the previously detected cloud.
   * Segments are represented as circles with a given center, radius, measure
   * and order parameter maximum value.
   */
  template <int dim>
  class Segment
  {
  public:
    Segment()
      : representation(nullptr)
    {}

    Segment(const Point<dim> &                center_in,
            const double                      radius_in,
            const double                      measure_in,
            const double                      max_value_in,
            std::unique_ptr<Representation> &&representation_in)
      : center(center_in)
      , radius(radius_in)
      , measure(measure_in)
      , max_value(max_value_in)
      , representation(std::move(representation_in))
    {}

    Segment(const Segment<dim> &other)
      : center(other.center)
      , radius(other.radius)
      , measure(other.measure)
      , max_value(other.max_value)
      , representation(other.representation ? other.representation->clone() :
                                              nullptr)
    {}

    const Point<dim> &
    get_center() const
    {
      return center;
    }

    double
    get_radius() const
    {
      return radius;
    }

    double
    get_measure() const
    {
      return measure;
    }


    double
    get_max_value() const
    {
      return max_value;
    }

    bool
    trivial() const
    {
      return !representation || representation->trivial();
    }

    double
    distance(const Segment<dim> &other) const
    {
      AssertThrow(
        representation,
        ExcMessage(
          "Representation should be initialized to compute the distance to a neighbor"));

      return representation->distance(*other.representation);
    }

    template <typename Stream>
    void
    print(Stream &stream) const
    {
      hpsint::print(*representation, stream);
      stream << " | max_value = " << max_value;
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &center;
      ar &radius;
      ar &measure;
      ar &max_value;
      ar &representation;
    }

  protected:
    Point<dim> center;

    double radius;
    double measure;
    double max_value;

    std::unique_ptr<Representation> representation;
  };
} // namespace GrainTracker