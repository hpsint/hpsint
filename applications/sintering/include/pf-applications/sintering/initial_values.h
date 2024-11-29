// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the hpsint authors
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

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace Sintering
{
  using namespace dealii;

  DeclException2(ExcMaxGrainsExceeded,
                 unsigned int,
                 unsigned int,
                 << "The initial condition contains too many order parameters: "
                 << arg1 << "; but only <= " << arg2 << " are allowed!"
                 << std::endl
                 << std::endl
                 << "Try to enable compressed initialization if the chosen"
                 << " initial values class provides this feature."
                 << std::endl);

  enum class InterfaceDirection
  {
    inside,
    outside,
    middle
  };

  inline InterfaceDirection
  to_interface_direction(const std::string label)
  {
    if (label == "Inside")
      return InterfaceDirection::inside;
    if (label == "Outside")
      return InterfaceDirection::outside;
    if (label == "Middle")
      return InterfaceDirection::middle;

    AssertThrow(false, ExcNotImplemented());

    return InterfaceDirection::middle;
  }

  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues(
      const double             interface_width,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset = 2,
      const bool               is_accumulative      = false)
      : Function<dim>(1)
      , interface_width(interface_width)
      , interface_direction(interface_direction)
      , op_components_offset(op_components_offset)
      , is_accumulative(is_accumulative)
      , current_component(numbers::invalid_unsigned_int)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      AssertDimension(component, 0);

      (void)component;

      return do_value(p, current_component);
    }

    virtual std::pair<Point<dim>, Point<dim>>
    get_domain_boundaries() const = 0;

    virtual double
    get_r_max() const = 0;

    double
    get_interface_width() const
    {
      return interface_width;
    }

    void
    set_component(const unsigned int current_component) const
    {
      AssertIndexRange(current_component, n_components());
      this->current_component = current_component;
    }

    unsigned int
    n_components() const
    {
      return n_order_parameters() + op_components_offset;
    }

    virtual unsigned int
    n_order_parameters() const = 0;

    virtual unsigned int
    n_particles() const = 0;

  protected:
    virtual double
    do_value(const Point<dim> &p, const unsigned int component) const = 0;

  protected:
    // Interface thickness
    const double interface_width;

    // Interface offset direction
    const InterfaceDirection interface_direction;

    // Offset for the order params offset
    const unsigned int op_components_offset;

    /* This parameter defines how particles interact within a grain boundary at
     * the initial configuration: whether the particles barely touch each other
     * or proto-necks are built up.
     *
     * That what happens at the grain boundary for the case of two particles:
     *    - false -> min(eta0, eta1)
     *    - true  -> eta0 + eta1
     */
    const bool is_accumulative;

  private:
    mutable unsigned int current_component;
  };
} // namespace Sintering
