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

// Sintering of N particles loaded from a CSV file

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

//#define WITH_TIMING
//#define WITH_TRACKER

#include "sintering/particle.h"
#include "sintering/sintering_impl.h"
#include "sintering/util.h"

/**
 * Custom message exception
 */
DeclException1(ExcCustomMessage, std::string, << arg1);

namespace Sintering
{
  template <int dim>
  class InitialValuesCloud : public InitialValues<dim>
  {
  private:
    struct Contact
    {
      unsigned int           primary;
      unsigned int           secondary;
      dealii::Tensor<2, dim> rotation_matrix;
      dealii::Tensor<1, dim> xm;
    };

    std::vector<Particle<dim>> particles;

    dealii::Point<dim> z_axis;

    dealii::Point<dim> ex0;
    dealii::Point<dim> ey0;
    dealii::Point<dim> ez0;

    std::map<std::pair<unsigned int, unsigned int>, Contact> contacts;

    double interface_width;

    double c_threshold = 0.5;

  public:
    InitialValuesCloud(const std::vector<Particle<dim>> &particles_in,
                       const double                      interface_width)
      : InitialValues<dim>(particles_in.size() + 2)
      , particles(particles_in)
      , interface_width(interface_width)
    {
      if (dim == 2)
        {
          ex0 = dealii::Point<dim>(1, 0);
          ey0 = dealii::Point<dim>(0, 1);
        }
      else if (dim == 3)
        {
          ex0 = dealii::Point<dim>(1, 0, 0);
          ey0 = dealii::Point<dim>(0, 1, 0);
          ez0 = dealii::Point<dim>(0, 0, 1);
        }
      else
        {
          throw std::runtime_error(
            "InitialValuesCloud not implemented for this dim space");
        }

      const unsigned int n_particles = particles.size();

      for (unsigned int i = 0; i < n_particles; i++)
        {
          auto &p1 = particles[i];

          for (unsigned int j = 0; j < n_particles; j++)
            {
              if (i != j)
                {
                  const auto &p2 = particles[j];

                  dealii::Tensor<1, dim> dir_vec  = p2.center - p1.center;
                  double                 distance = dir_vec.norm();
                  double                 r1       = p1.radius;
                  double                 r2       = p2.radius;
                  dealii::Tensor<1, dim> ex       = dir_vec / distance;

                  double dist_max = r1 + r2;

                  if (distance <= dist_max)
                    {
                      double s = (r1 + r2 + distance) / 2;
                      double r0 =
                        2 / distance *
                        std::sqrt(s * (s - r1) * (s - r2) * (s - distance));

                      double dm = std::sqrt(r1 * r1 - r0 * r0);

                      dealii::Tensor<1, dim> ey;
                      if (dim == 2)
                        {
                          ey = dealii::cross_product_2d(ex);
                        }
                      else if (dim == 3)
                        {
                          dealii::Tensor<1, dim> z_temp = z_axis - p1.center;
                          ey = dealii::cross_product_3d(z_temp, ex);
                          ey /= ey.norm();
                        }

                      dealii::Tensor<2, dim> rotation_matrix =
                        dealii::outer_product(ex, ex0) +
                        dealii::outer_product(ey, ey0);
                      if (dim == 3)
                        {
                          dealii::Tensor<1, dim> ez(
                            dealii::cross_product_3d(ex, ey));
                          rotation_matrix += dealii::outer_product(ez, ez0);
                        }

                      dealii::Tensor<1, dim> xm = p1.center + dm * ex;

                      auto    key = std::make_pair(i, j);
                      Contact c{p1.id, p2.id, rotation_matrix, xm};

                      contacts[key] = c;

                      p1.neighbours.push_back(j);
                    }
                }
            }
        }
    }

    virtual double
    value(const dealii::Point<dim> &p,
          const unsigned int        component = 0) const override
    {
      double ret_val = 0;

      if (component == 0)
        {
          double c_main = 0;

          for (const auto &particle_current : particles)
            {
              double c_current = this->is_in_sphere(p,
                                                    particle_current.center,
                                                    particle_current.radius);

              c_main = std::max(c_main, c_current);
            }

          ret_val = c_main;
        }
      else if (component == 1)
        {
          ret_val = 0;
        }
      else
        {
          unsigned int i = component - 2;

          const auto &particle_current = particles[i];

          double c_main = this->is_in_sphere(p,
                                             particle_current.center,
                                             particle_current.radius);

          double       c_secondary  = 0;
          unsigned int secondary_id = std::numeric_limits<unsigned int>::max();
          for (unsigned int j : particle_current.neighbours)
            {
              const auto &particle_secondary = particles[j];

              double c_current = this->is_in_sphere(p,
                                                    particle_secondary.center,
                                                    particle_secondary.radius);

              if (c_current > c_secondary)
                {
                  secondary_id = particle_secondary.id;
                  c_secondary  = c_current;
                }
            }

          if (secondary_id < std::numeric_limits<unsigned int>::max())
            {
              auto key = std::make_pair(particle_current.id, secondary_id);
              const auto &           contact = contacts.at(key);
              dealii::Tensor<1, dim> dm      = p - contact.xm;

              dealii::Tensor<1, dim> dm_local = contact.rotation_matrix * dm;

              double xp = dm_local[0];

              if (xp < interface_width / 2.0)
                {
                  if (c_secondary > c_threshold)
                    {
                      if (xp < interface_width / 2.0 &&
                          xp > -interface_width / 2.0)
                        {
                          c_main =
                            0.5 - 0.5 * std::sin(M_PI * xp / interface_width);
                        }
                    }
                }
              else
                {
                  c_main = 0;
                }
            }

          ret_val = c_main;
        }

      return ret_val;
    }

    virtual std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const override
    {
      std::pair<dealii::Point<dim>, dealii::Point<dim>> boundaries;

      for (unsigned int i = 0; i < dim; i++)
        {
          boundaries.first[i]  = std::numeric_limits<double>::max();
          boundaries.second[i] = std::numeric_limits<double>::min();
        }

      for (const auto &p : particles)
        {
          for (unsigned int i = 0; i < dim; i++)
            {
              if (p.center[i] - p.radius < boundaries.first[i])
                {
                  boundaries.first[i] = p.center[i] - p.radius;
                }
              if (p.center[i] + p.radius > boundaries.second[i])
                {
                  boundaries.second[i] = p.center[i] + p.radius;
                }
            }
        }

      return boundaries;
    }

    virtual double
    get_r_max() const override
    {
      const auto &pt_rmax = std::max_element(particles.begin(),
                                             particles.end(),
                                             [](const auto &a, const auto &b) {
                                               return a.radius < b.radius;
                                             });
      return pt_rmax->radius;
    }

    virtual double
    get_interface_width() const override
    {
      return interface_width;
    }
  };
} // namespace Sintering

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;
  std::string           file_cloud;

  if (argc == 2)
    {
      if (std::string(argv[1]) == "--help")
        {
          dealii::ConditionalOStream pcout(
            std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

          pcout << "Arguments list: cloud_file [param_file]" << std::endl;
          pcout
            << "    cloud_file - a path to a CSV file containing particles list"
            << std::endl;
          pcout << "    param_file - a path to a prm file" << std::endl;

          pcout << std::endl;
          params.print();
          return 0;
        }
      else
        {
          file_cloud = std::string(argv[1]);
        }
    }
  else if (argc == 3)
    {
      file_cloud = std::string(argv[1]);
      params.parse(std::string(argv[2]));
    }
  else
    {
      throw std::runtime_error("Argument cloud_file has to be provided");
    }

  std::ifstream fstream(file_cloud.c_str());

  auto particles = Sintering::read_particles<SINTERING_DIM>(fstream);

  AssertThrow(particles.size() == SINTERING_GRAINS,
              ExcCustomMessage(
                "The CSV file contains wrong number of particles: " +
                std::to_string(particles.size()) + " but has to be " +
                std::to_string(SINTERING_GRAINS)));

  // geometry
  static constexpr double interface_width = 2.0;

  auto initial_solution =
    std::make_shared<Sintering::InitialValuesCloud<SINTERING_DIM>>(
      particles, interface_width);

  Sintering::Problem<SINTERING_DIM, SINTERING_GRAINS> runner(params,
                                                             initial_solution);
  runner.run();
}
