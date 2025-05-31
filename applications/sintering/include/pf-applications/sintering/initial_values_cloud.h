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

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <pf-applications/sintering/initial_values_spherical.h>
#include <pf-applications/sintering/particle.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim>
  class InitialValuesCloud : public InitialValuesSpherical<dim>
  {
  public:
    InitialValuesCloud(
      const std::vector<Particle<dim>> &particles_in,
      const double                      interface_width,
      const bool                        minimize_order_parameters,
      const double                      interface_buffer_ratio = 0.5,
      const double                      radius_buffer_ratio    = 0.0,
      const InterfaceDirection interface_direction = InterfaceDirection::middle,
      const unsigned int       op_components_offset  = 2,
      const bool               concentration_as_void = false,
      const bool               is_accumulative       = false)
      : InitialValuesSpherical<dim>(interface_width,
                                    interface_direction,
                                    op_components_offset,
                                    concentration_as_void,
                                    is_accumulative)
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

      // DSP for colorization if order parameters are compressed
      DynamicSparsityPattern dsp(n_particles);

      // Detect contacts
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
                  if (distance <= dist_max - 1e-3 * dist_max)
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
                        dealii::outer_product(ex0, ex) +
                        dealii::outer_product(ey0, ey);
                      if (dim == 3)
                        {
                          const dealii::Point<dim> ez(
                            dealii::cross_product_3d(ex, ey));
                          rotation_matrix += dealii::outer_product(ez0, ez);
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

                  /* We also prepare for colorization if order parameters are
                   * compressed.
                   */
                  if (minimize_order_parameters)
                    {
                      double buffer = interface_width +
                                      interface_buffer_ratio * interface_width +
                                      radius_buffer_ratio * std::max(r1, r2);
                      if (distance <= dist_max + buffer)
                        {
                          dsp.add(i, j);
                        }
                    }
                }
            }
        }

      // Build colorization if compressed
      if (minimize_order_parameters)
        {
          SparsityPattern sp;
          sp.copy_from(dsp);

          std::vector<unsigned int> color_indices(n_particles);
          SparsityTools::color_sparsity_pattern(sp, color_indices);

          for (unsigned int i = 0; i < n_particles; i++)
            {
              order_parameter_to_grains[color_indices[i] - 1].push_back(i);
            }
        }
      else
        {
          for (unsigned int i = 0; i < n_particles; i++)
            {
              order_parameter_to_grains[i].push_back(i);
            }
        }
      order_parameters_num = order_parameter_to_grains.size();
    }

    double
    op_value(const dealii::Point<dim> &p,
             const unsigned int        order_parameter) const final
    {
      AssertThrow(order_parameter < n_order_parameters(),
                  ExcMessage(
                    "Incorrect order parameter " +
                    std::to_string(order_parameter) +
                    " requested, but the total number of order parameters is " +
                    std::to_string(n_order_parameters())));

      double ret_val = 0;

      for (const auto pid : order_parameter_to_grains.at(order_parameter))
        {
          const auto &particle_current = particles.at(pid);
          ret_val = value_for_particle(p, particle_current);

          if (ret_val != 0)
            break;
        }

      return ret_val;
    }

    std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const final
    {
      return boundaries;
    }

    std::map<unsigned int, std::vector<unsigned int>>
    get_order_parameter_to_grains() const
    {
      return order_parameter_to_grains;
    }

    double
    get_r_max() const final
    {
      const auto &pt_rmax = std::max_element(particles.begin(),
                                             particles.end(),
                                             [](const auto &a, const auto &b) {
                                               return a.radius < b.radius;
                                             });
      return pt_rmax->radius;
    }

    unsigned int
    n_order_parameters() const final
    {
      return order_parameters_num;
    }

    unsigned int
    n_contacts() const
    {
      return contacts.size();
    }

    unsigned int
    n_particles() const final
    {
      return particles.size();
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

    // Number of order parameters in use
    unsigned int order_parameters_num;

    // Map order parameters to specific grains
    std::map<unsigned int, std::vector<unsigned int>> order_parameter_to_grains;

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

    double
    value_for_particle(const Point<dim>    &p,
                       const Particle<dim> &particle_current) const
    {
      // Concentration value for the i-th particle
      double c_main =
        this->is_in_sphere(p, particle_current.center, particle_current.radius);

      if (c_main != 0)
        {
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
        }

      return c_main;
    }

    template <typename Stream>
    void
    print_contacts(Stream &stream) const
    {
      for (const auto &[from_to, contact] : contacts)
        {
          stream << from_to.first << "->" << from_to.second << ": "
                 << "primary = " << contact.primary
                 << ", secondary = " << contact.secondary
                 << ", center = " << contact.center
                 << ", rotation = " << contact.rotation_matrix << std::endl;
        }
    }
  };
} // namespace Sintering