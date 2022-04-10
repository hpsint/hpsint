// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Sintering of N particles located along the circle

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

//#define WITH_TIMING
//#define WITH_TRACKER

#include <pf-applications/sintering/driver.h>

using namespace dealii;

namespace Sintering
{
  template <int dim>
  class InitialValuesCircle : public InitialValues<dim>
  {
  public:
    InitialValuesCircle(const double       r0,
                        const double       interface_width,
                        const unsigned int n_grains,
                        const bool         minimize_order_parameters,
                        const bool         is_accumulative)
      : InitialValues<dim>()
      , r0(r0)
      , interface_width(interface_width)
      , is_accumulative(is_accumulative)
    {
      const double alfa = 2 * M_PI / n_grains;

      const double h = r0 / std::sin(alfa / 2.);

      for (unsigned int ip = 0; ip < n_grains; ip++)
        {
          std::array<double, dim> scoords{{h, ip * alfa}};
          centers.push_back(
            dealii::GeometricUtilities::Coordinates::from_spherical<dim>(
              scoords));
        }

      if (minimize_order_parameters)
        {
          if (n_grains == 1)
            {
              order_parameter_to_grains[0];
            }
          else
            {
              order_parameter_to_grains[0];
              order_parameter_to_grains[1];

              for (unsigned int ip = 0; ip < n_grains; ip++)
                {
                  const unsigned int current_order_parameter = ip % 2;
                  order_parameter_to_grains.at(current_order_parameter)
                    .push_back(ip);
                }

              /* If the number of particles is odd, then the order parameter of
               * the last grain has to be changed to 2
               */
              if (n_grains % 2)
                {
                  const unsigned int last_grain =
                    order_parameter_to_grains.at(0).back();
                  order_parameter_to_grains.at(0).pop_back();

                  order_parameter_to_grains[2] = {last_grain};
                }
            }
        }
      else
        {
          for (unsigned int ip = 0; ip < n_grains; ip++)
            {
              order_parameter_to_grains[ip] = {ip};
            }
        }
    }

    double
    do_value(const dealii::Point<dim> &p,
             const unsigned int        component) const final
    {
      double ret_val = 0;

      if (component == 0)
        {
          std::vector<double> etas;
          for (const auto &pt : centers)
            {
              etas.push_back(this->is_in_sphere(p, pt, r0));
            }

          if (is_accumulative)
            {
              ret_val = std::accumulate(etas.begin(),
                                        etas.end(),
                                        0,
                                        [](auto a, auto b) {
                                          return std::move(a) + b;
                                        });
              if (ret_val > 1.0)
                {
                  ret_val = 1.0;
                }
            }
          else
            {
              ret_val = *std::max_element(etas.begin(), etas.end());
            }
        }
      else if (component == 1)
        {
          ret_val = 0;
        }
      else
        {
          const unsigned int order_parameter = component - 2;

          for (const auto pid : order_parameter_to_grains.at(order_parameter))
            {
              const auto &pt = centers.at(pid);
              ret_val        = this->is_in_sphere(p, pt, r0);

              if (ret_val != 0)
                {
                  break;
                }
            }
        }

      return ret_val;
    }

    std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const final
    {
      const auto pt_xmax = *std::max_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[0] < b[0];
                                             });

      const auto pt_ymax = *std::max_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[1] < b[1];
                                             });

      const auto pt_xmin = *std::min_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[0] < b[0];
                                             });

      const auto pt_ymin = *std::min_element(centers.begin(),
                                             centers.end(),
                                             [](const auto &a, const auto &b) {
                                               return a[1] < b[1];
                                             });

      double xmin = pt_xmin[0] - r0;
      double xmax = pt_xmax[0] + r0;
      double ymin = pt_ymin[1] - r0;
      double ymax = pt_ymax[1] + r0;

      if (dim == 2)
        {
          return std::make_pair(dealii::Point<dim>(xmin, ymin),
                                dealii::Point<dim>(xmax, ymax));
        }
      else if (dim == 3)
        {
          double zmin = -r0;
          double zmax = r0;
          return std::make_pair(dealii::Point<dim>(xmin, ymin, zmin),
                                dealii::Point<dim>(xmax, ymax, zmax));
        }
    }

    double
    get_r_max() const final
    {
      return r0;
    }

    double
    get_interface_width() const final
    {
      return interface_width;
    }

    unsigned int
    n_order_parameters() const final
    {
      return order_parameter_to_grains.size();
    }

  private:
    std::vector<dealii::Point<dim>> centers;

    double r0;
    double interface_width;

    /* This parameter defines how particles interact within a grain boundary at
     * the initial configuration: whether the particles barely touch each other
     * or proto-necks are built up.
     *
     * That what happens at the grain boundary for the case of two particles:
     *    - false -> min(eta0, eta1)
     *    - true  -> eta0 + eta1
     */
    bool is_accumulative;

    // Map order parameters to specific grains
    std::map<unsigned int, std::vector<unsigned int>> order_parameter_to_grains;
  };
} // namespace Sintering

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;

  if (argc == 2 && std::string(argv[1]) == "--help")
    {
      params.print();
      return 0;
    }

  AssertThrow(2 <= argc && argc <= 3, ExcNotImplemented());

  const unsigned int n_grains = atoi(argv[1]);

  if (argc == 3)
    params.parse(std::string(argv[2]));

  // geometry
  static constexpr double r0              = 15.0 / 2.;
  static constexpr bool   is_accumulative = false;

  const auto initial_solution =
    std::make_shared<Sintering::InitialValuesCircle<SINTERING_DIM>>(
      r0,
      params.geometry_data.interface_width,
      n_grains,
      params.geometry_data.minimize_order_parameters,
      is_accumulative);

  AssertThrow(initial_solution->n_order_parameters() <= MAX_SINTERING_GRAINS,
              Sintering::ExcMaxGrainsExceeded(
                initial_solution->n_order_parameters(), MAX_SINTERING_GRAINS));

  Sintering::Problem<SINTERING_DIM> runner(params, initial_solution);
  runner.run();
}
