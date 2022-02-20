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

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

//#define WITH_TIMING
//#define WITH_TRACKER

#include <pf-applications/sintering/driver.h>
#include <pf-applications/sintering/particle.h>

#include <cstdlib>

using namespace dealii;

namespace Sintering
{
  template <int dim>
  class InitialValuesCloud : public InitialValues<dim>
  {
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
          AssertThrow(
            false,
            ExcMessage(
              "InitialValuesCloud not implemented for this dim space"));
        }

      // Compute domain boundaries
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

      const unsigned int n_particles = particles.size();

      for (unsigned int i = 0; i < n_particles; i++)
        {
          auto &p1 = particles[i];

          for (unsigned int j = 0; j < n_particles; j++)
            {
              if (i != j)
                {
                  const auto &p2 = particles[j];

                  // Compute directional vector from particle 1 to particle 2 of
                  // the contact pair, this defines local x-axis and the
                  // corresponding unitary vector ex
                  const dealii::Point<dim> dir_vec(p2.center - p1.center);
                  const double             distance = dir_vec.norm();
                  const dealii::Point<dim> ex       = dir_vec / distance;

                  const double r1 = p1.radius;
                  const double r2 = p2.radius;

                  // The maximum possible distance between the centers of the
                  // particles
                  const double dist_max = r1 + r2;

                  // If distance is smaller than the maximum value, then we have
                  // an overlap
                  if (distance <= dist_max)
                    {
                      // Compute the radius of the neck
                      const double s = (r1 + r2 + distance) / 2;
                      const double r0 =
                        2 / distance *
                        std::sqrt(s * (s - r1) * (s - r2) * (s - distance));

                      // Distance from the center of particle 1 to the center of
                      // the neck
                      const double dm = std::sqrt(r1 * r1 - r0 * r0);

                      // Compute unitary vector ey of the y-axis
                      dealii::Point<dim> ey;
                      if (dim == 2)
                        {
                          ey = dealii::cross_product_2d(ex);
                        }
                      else if (dim == 3)
                        {
                          dealii::Point<dim> current_z_orientation =
                            get_orientation_point(p1.center, ex);
                          dealii::Point<dim> z_temp(current_z_orientation -
                                                    p1.center);

                          ey = dealii::cross_product_3d(z_temp, ex);
                          ey /= ey.norm();
                        }

                      // Build up the rotation matrix of the local coordinate
                      // system
                      dealii::Tensor<2, dim> rotation_matrix =
                        dealii::outer_product(ex, ex0) +
                        dealii::outer_product(ey, ey0);
                      if (dim == 3)
                        {
                          const dealii::Point<dim> ez(
                            dealii::cross_product_3d(ex, ey));
                          rotation_matrix += dealii::outer_product(ez, ez0);
                        }

                      // Coordinate of the central contact point in global
                      // coordinates
                      const dealii::Point<dim> contact_center =
                        p1.center + dm * ex;

                      // Create a new contact pair
                      auto key = std::make_pair(i, j);

                      const Contact c{p1.id,
                                      p2.id,
                                      rotation_matrix,
                                      contact_center};

                      contacts[key] = c;

                      // Add j-th neighbour for the i-th particle
                      p1.neighbours.push_back(j);
                    }
                }
            }
        }
    }

    double
    do_value(const dealii::Point<dim> &p,
             const unsigned int        component = 0) const final
    {
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

          return c_main;
        }
      else if (component == 1)
        {
          return 0;
        }
      else
        {
          AssertThrow(component < this->n_components(),
                      ExcMessage(
                        "Incorrect function component " +
                        std::to_string(component) +
                        " requested, but the total number of components is " +
                        std::to_string(this->n_components())));

          unsigned int i = component - 2;

          AssertThrow(
            i < particles.size(),
            ExcMessage(
              "Incorrect particle i = " + std::to_string(i) +
              " requested, but the total number of particles in the domain is " +
              std::to_string(particles.size())));

          const auto &particle_current = particles[i];

          // Concentration value for the i-th particle
          double c_main = this->is_in_sphere(p,
                                             particle_current.center,
                                             particle_current.radius);


          /* At the current point 'p' multiple particles may overlap thus they
           * all will have some value of concentration > 0. We go through all
           * the neighbours of the current i-th particle and find the neighbour
           * whose concentration is higher than the other neighbours. Such a
           * particle is said to be secondary.
           */
          double       c_secondary  = 0;
          unsigned int secondary_id = numbers::invalid_unsigned_int;
          for (unsigned int j : particle_current.neighbours)
            {
              const auto &particle_secondary_candidate = particles[j];

              double c_secondary_candidate =
                this->is_in_sphere(p,
                                   particle_secondary_candidate.center,
                                   particle_secondary_candidate.radius);

              if (c_secondary_candidate > c_secondary)
                {
                  secondary_id = particle_secondary_candidate.id;
                  c_secondary  = c_secondary_candidate;
                }
            }

          /* If we have found a secondary particle, then make some additional
           * steps. If point 'p' is somewhere deep inside the particle, then non
           * of the neighbours contributes its concentration and we skip this
           * part.
           */
          if (secondary_id < numbers::invalid_unsigned_int)
            {
              // Extract the contact object for the two particles
              auto key = std::make_pair(particle_current.id, secondary_id);
              const auto &contact = contacts.at(key);

              /* Vector from the contact neck center to the current point 'p' in
               * global coordinate system
               */
              auto dm_global = p - contact.center;

              /* Vector from the contact neck center to the current point 'p' in
               * local coordinate system
               */
              auto dm_local = contact.rotation_matrix * dm_global;

              // Get x-axis component of the vector
              double xp = dm_local[0];

              /* This block constructs a smooth transient interface over the
               * neck (chord) between the 2 particles. Since the interface is
               * diffuse, the zone of interest lies a bit outside of the contact
               * area.
               */
              if (xp < interface_width / 2.0)
                {
                  /* If the concentration of secondary particle is above the
                   * threshold, it means that point 'p' is at the chord between
                   * the two particles and we need to build an interface which
                   * is simply a straight line for 2D or plane for 3D. The new
                   * interfaces is diffuse along the local x-axis of the contact
                   * pair.
                   */
                  if (c_secondary > c_threshold && xp > -interface_width / 2.0)
                    {
                      /* The new interfaces is diffusive along the local x-axis
                       * of the contact pair.
                       */
                      c_main = 0.5 - 0.5 * std::sin(numbers::PI * xp /
                                                    interface_width);
                    }
                }
              else
                {
                  /* This means that point 'p' lies inside that part of the main
                   * particle which is cut out but the secondary neighbour, so
                   * the concentration contribution of the main particle
                   * vanishes at this point of the domain.
                   */
                  c_main = 0;
                }
            }

          return c_main;
        }
    }

    virtual std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const override
    {
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

  private:
    // A special struct that stores contact infromation for a pair of particles
    struct Contact
    {
      // id of the primary (first) particle
      unsigned int primary;

      // id of the primary (first) particle
      unsigned int secondary;

      // Rotation matrix of the local coordinate system.
      dealii::Tensor<2, dim> rotation_matrix;

      // Coordinate of the central point of the contact edge between the two
      // particles in the global coordinate system
      dealii::Point<dim> center;
    };

    std::vector<Particle<dim>> particles;

    // Oriental point for building z-axis (in case of 3D)
    dealii::Point<dim> z_axis_orientation;

    // Unitary vectors of the global coordinate system
    dealii::Point<dim> ex0;
    dealii::Point<dim> ey0;
    dealii::Point<dim> ez0;

    std::map<std::pair<unsigned int, unsigned int>, Contact> contacts;

    double interface_width;

    /* Threshold value for detecting the secondary grain at the contact
     * neighbourhood between the 2 particles. If the value is below the
     * threshold, then the secondary grain does not contribute to the primary
     * one any longer. */
    double c_threshold = 0.5;

    // Domain boundaries
    std::pair<dealii::Point<dim>, dealii::Point<dim>> boundaries;

    double
    drand(double dmin, double dmax) const
    {
      double val = (double)std::rand() / RAND_MAX;
      return dmin + val * (dmax - dmin);
    }

    bool
    check_orientation_point(
      const dealii::Point<dim> &origin,
      const dealii::Point<dim> &ex,
      const dealii::Point<dim> &candidate_z_orientation) const
    {
      const double tol = 1e-12;

      dealii::Point<dim> ez_temp(candidate_z_orientation - origin);

      if (ez_temp.norm() < tol)
        {
          return false;
        }

      ez_temp /= ez_temp.norm();

      if (std::abs(ex * ez_temp - 1) < tol)
        {
          return false;
        }

      return true;
    }

    dealii::Point<dim>
    get_orientation_point(const dealii::Point<dim> &origin,
                          const dealii::Point<dim> &ex) const
    {
      /* The default z-axis orientation point is (0,0,0). It may not fit if
       * either of the particles coincides with this point or it turns out to be
       * colinear with the x-axis of either of the contact pairs. If such a
       * situation is detected, we then need to choose another point, randomly
       * selected within the boundaries of the domain.
       */

      dealii::Point<dim> current_orientation = z_axis_orientation;

      bool is_valid = check_orientation_point(origin, ex, current_orientation);

      if (!is_valid)
        {
          unsigned int iter_max = 100;

          for (unsigned int i = 0; i < iter_max && !is_valid; i++)
            {
              for (unsigned int j = 0; j < dim; j++)
                {
                  current_orientation[j] =
                    drand(boundaries.first[j], boundaries.second[j]);
                }
              is_valid =
                check_orientation_point(origin, ex, current_orientation);
            }
        }

      AssertThrow(is_valid,
                  ExcMessage("Failed to generate valid orientation point"));

      return current_orientation;
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
      AssertThrow(false, ExcMessage("Argument cloud_file has to be provided"));
    }

  std::ifstream fstream(file_cloud.c_str());

  const auto particles = Sintering::read_particles<SINTERING_DIM>(fstream);

  AssertThrow(particles.size() <= MAX_SINTERING_GRAINS,
              ExcMessage("The CSV file contains wrong number of particles: " +
                         std::to_string(particles.size()) +
                         " but has to be leq" +
                         std::to_string(MAX_SINTERING_GRAINS)));

  // geometry
  static constexpr double interface_width = 2.0;

  const auto initial_solution =
    std::make_shared<Sintering::InitialValuesCloud<SINTERING_DIM>>(
      particles, interface_width);

  Sintering::Problem<SINTERING_DIM> runner(params, initial_solution);
  runner.run();
}
