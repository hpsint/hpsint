#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace Sintering
{
  using namespace dealii;

  DeclException2(ExcMaxGrainsExceeded,
                 unsigned int,
                 unsigned int,
                 << "The initial conditions contain too many order parameters:"
                 << arg1 << "but has to be <= " << arg2 << std::endl
                 << std::endl
                 << "Try to enable compressed initialization if the chosen"
                 << " initial values class provides this feature."
                 << std::endl);

  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
    InitialValues(double interface_offset = 0)
      : Function<dim>(1)
      , current_component(numbers::invalid_unsigned_int)
      , interface_offset(interface_offset)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const final
    {
      AssertDimension(component, 0);

      (void)component;

      return this->do_value(p, current_component);
    }

    virtual std::pair<Point<dim>, Point<dim>>
    get_domain_boundaries() const = 0;

    virtual double
    get_r_max() const = 0;

    virtual double
    get_interface_width() const = 0;

    void
    set_component(const unsigned int current_component)
    {
      AssertIndexRange(current_component, n_components());
      this->current_component = current_component;
    }

    unsigned int
    n_components() const
    {
      return n_order_parameters() + 2;
    }

    virtual unsigned int
    n_order_parameters() const = 0;

  private:
    unsigned int current_component;
    const double interface_offset;

  protected:
    virtual double
    do_value(const Point<dim> &p, const unsigned int component) const = 0;

    double
    is_in_sphere(const Point<dim> &point,
                 const Point<dim> &center,
                 double            rc) const
    {
      double c = 0;

      double rm  = rc - interface_offset;
      double rad = center.distance(point);

      if (rad <= rm - get_interface_width() / 2.0)
        {
          c = 1;
        }
      else if (rad < rm + get_interface_width() / 2.0)
        {
          double outvalue = 0.;
          double invalue  = 1.;
          double int_pos =
            (rad - rm + get_interface_width() / 2.0) / get_interface_width();

          c = outvalue + (invalue - outvalue) *
                           (1.0 + std::cos(int_pos * numbers::PI)) / 2.0;
        }

      return c;
    }
  };
} // namespace Sintering