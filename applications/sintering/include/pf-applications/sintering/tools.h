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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <pf-applications/base/tensor.h>

namespace Sintering
{
  using namespace dealii;

  enum class InitialRefine
  {
    None,
    Base,
    Full
  };

  std::vector<unsigned int>
  get_primes(unsigned int start, unsigned int end)
  {
    std::vector<unsigned int> primes;

    for (unsigned int i = start; i <= end; ++i)
      {
        // Skip 0 and 1
        if (i == 1 || i == 0)
          continue;

        bool is_prime = true;

        // Iterate to check if i is prime
        for (unsigned int j = 2; j <= i / 2; ++j)
          if (i % j == 0)
            {
              is_prime = false;
              break;
            }

        if (is_prime)
          primes.push_back(i);
      }
    return primes;
  }

  std::pair<unsigned int, unsigned int>
  decompose_to_prime_tuple(const unsigned int n_ref, const unsigned max_prime)
  {
    const auto primes = get_primes(2, max_prime);

    unsigned int optimal_prime      = 0;
    unsigned int n_refinements      = 0;
    unsigned int min_elements_delta = numbers::invalid_unsigned_int;
    for (const auto &p : primes)
      {
        const unsigned int s =
          static_cast<unsigned int>(std::ceil(std::log2(n_ref / p)));
        const unsigned int n_current     = p * std::pow(2, s);
        const unsigned int current_delta = static_cast<unsigned int>(
          std::abs(static_cast<int>(n_current - n_ref)));

        if (current_delta < min_elements_delta)
          {
            min_elements_delta = current_delta;
            optimal_prime      = p;
            n_refinements      = s;
          }
      }

    return std::make_pair(optimal_prime, n_refinements);
  }

  template <int dim>
  void
  print_mesh_info(const Point<dim> &               bottom_left,
                  const Point<dim> &               top_right,
                  const std::vector<unsigned int> &subdivisions,
                  const unsigned int               n_refinements_global,
                  const unsigned int               n_refinements_delayed)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    pcout << "Create subdivided hyperrectangle [";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(top_right[i] - bottom_left[i]);

        if (i + 1 != dim)
          pcout << " x ";
      }
    pcout << "] = [";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(bottom_left[i]) << "..."
              << std::to_string(top_right[i]);

        if (i + 1 != dim)
          pcout << " x ";
      }
    pcout << "] " << std::endl;

    const unsigned int n_refinements =
      n_refinements_global + n_refinements_delayed;

    pcout << "with " << std::to_string(n_refinements) << " refinements (";
    pcout << "global = " << n_refinements_global << ", ";
    pcout << "delayed = " << n_refinements_delayed << ") and ";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(subdivisions[i]);

        if (i + 1 != dim)
          pcout << "x";
      }

    pcout << " subdivisions" << std::endl << std::endl;
  }

  namespace internal
  {
    unsigned int
    adjust_divisions_to_primes(const unsigned int         max_prime,
                               std::vector<unsigned int> &subdivisions)
    {
      // Further reduce the number of initial subdivisions
      unsigned int n_refinements_base = 0;
      if (max_prime > 0)
        {
          const unsigned int n_ref =
            *std::min_element(subdivisions.begin(), subdivisions.end());

          const auto         pair = decompose_to_prime_tuple(n_ref, max_prime);
          const unsigned int optimal_prime = pair.first;

          n_refinements_base = pair.second;

          for (unsigned int d = 0; d < dim; d++)
            {
              subdivisions[d] = static_cast<unsigned int>(std::ceil(
                static_cast<double>(subdivisions[d]) / n_ref * optimal_prime));
            }
        }

      return n_refinements_base;
    }

    template <int dim, typename Triangulation>
    void
    impose_periodicity(Triangulation &tria)
    {
      // Need to work with triangulation here
      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(
            tria, 2 * d, 2 * d + 1, d, periodicity_vector);
        }

      tria.add_periodicity(periodicity_vector);
    }

    template <typename Triangulation>
    std::pair<unsigned int, unsigned int>
    make_initial_refines(const InitialRefine refine,
                         const unsigned int  n_refinements_base,
                         const unsigned int  n_refinements_interface)
    {
      unsigned int n_global  = 0;
      unsigned int n_delayed = 0;
      if (refine == InitialRefine::Base)
        {
          tria.refine_global(n_refinements_base);

          n_global  = n_refinements_base;
          n_delayed = n_refinements_interface;
        }
      else if (refine == InitialRefine::Full)
        {
          tria.refine_global(n_refinements_base + n_refinements_interface);

          n_global  = n_refinements_base + n_refinements_interface;
          n_delayed = 0;
        }
      else
        {
          n_global  = 0;
          n_delayed = n_refinements_base + n_refinements_interface;
        }

      return {n_global, n_delayed};
    }
  } // namespace internal

  template <typename Triangulation, int dim>
  unsigned int
  create_mesh_from_interface(
    Triangulation &     tria,
    const Point<dim> &  bottom_left,
    const Point<dim> &  top_right,
    const double        interface_width,
    const double        divisions_per_interface,
    const bool          periodic,
    const InitialRefine refine,
    const unsigned int  max_prime                          = 0,
    const double        max_level0_divisions_per_interface = 1.0,
    const unsigned int  divisions_per_element              = 1,
    const bool          print_stats                        = true)
  {
    // Domain size
    const auto domain_size = top_right - bottom_left;

    // Recompute divisions to elements
    const double elements_per_interface =
      static_cast<double>(divisions_per_interface) / divisions_per_element;
    const double max_level0_elements_per_interface =
      max_level0_divisions_per_interface / divisions_per_element;

    // Desirable smallest element size
    const double h_e = interface_width / elements_per_interface;

    // Number of refinements to get the desirable element size
    const unsigned int n_refinements_interface =
      static_cast<unsigned int>(std::ceil(
        std::log2(elements_per_interface / max_level0_elements_per_interface)));

    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int d = 0; d < dim; d++)
      {
        subdivisions[d] = static_cast<unsigned int>(std::ceil(
          domain_size[d] / h_e / std::pow(2, n_refinements_interface)));
      }

    // Further reduce the number of initial subdivisions
    unsigned int n_refinements_base =
      internal::adjust_divisions_to_primes(max_prime, subdivisions);

    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      internal::impose_periodicity<dim>(tria);

    const auto [n_global, n_delayed] =
      internal::make_initial_refines(refine,
                                     n_refinements_base,
                                     n_refinements_interface);

    if (print_stats)
      print_mesh_info(
        bottom_left, top_right, subdivisions, n_global, n_delayed);

    return n_delayed;
  }

  template <typename Triangulation, int dim>
  unsigned int
  create_mesh_from_divisions(Triangulation &                  tria,
                             const Point<dim> &               bottom_left,
                             const Point<dim> &               top_right,
                             const std::vector<unsigned int> &subdivisions,
                             const bool                       periodic,
                             const unsigned int               n_refinements,
                             const bool print_stats = true)
  {
    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      internal::impose_periodicity<dim>(tria);

    tria.refine_global(n_refinements);

    if (print_stats)
      print_mesh_info(bottom_left, top_right, subdivisions, n_refinements, 0);

    // Return 0 delayed lazy refinements for consistency of the interfaces
    return 0;
  }

  namespace internal
  {
    template <typename VectorType, typename Triangulation, int dim>
    void
    prepare_coarsening_and_refinement(
      const VectorType &                    solution_to_estimate,
      Triangulation &                       tria,
      DoFHandler<dim> &                     dof_handler,
      const Quadrature<dim - 1> &           quad,
      const double                          top_fraction_of_cells,
      const double                          bottom_fraction_of_cells,
      const unsigned int                    min_allowed_level,
      const unsigned int                    max_allowed_level,
      const typename VectorType::value_type val_min = 0.05,
      const typename VectorType::value_type val_max = 0.95)
    {
      // estimate errors
      Vector<float> estimated_error_per_cell(tria.n_active_cells());

      for (unsigned int b = 0; b < solution_to_estimate.n_blocks(); ++b)
        {
          Vector<float> estimated_error_per_cell_temp(tria.n_active_cells());

          KellyErrorEstimator<dim>::estimate(
            dof_handler,
            quad,
            std::map<types::boundary_id, const Function<dim> *>(),
            solution_to_estimate.block(b),
            estimated_error_per_cell_temp,
            {},
            nullptr,
            0,
            Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

          for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
            estimated_error_per_cell[i] += estimated_error_per_cell_temp[i] *
                                           estimated_error_per_cell_temp[i];
        }

      for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
        estimated_error_per_cell[i] = std::sqrt(estimated_error_per_cell[i]);

      // mark automatically cells for coarsening/refinement, ...
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        tria,
        estimated_error_per_cell,
        top_fraction_of_cells,
        bottom_fraction_of_cells);

      // make sure that cells close to the interfaces are refined, ...
      Vector<typename VectorType::value_type> values(
        dof_handler.get_fe().n_dofs_per_cell());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false || cell->refine_flag_set())
            continue;

          for (unsigned int b = 0; b < solution_to_estimate.n_blocks(); ++b)
            {
              cell->get_dof_values(solution_to_estimate.block(b), values);

              typename VectorType::value_type val_avg = 0;

              for (unsigned int i = 0; i < values.size(); ++i)
                {
                  val_avg += values[i];

                  if (val_min < values[i] && values[i] < val_max)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();

                      break;
                    }
                }

              if (!cell->refine_flag_set())
                {
                  // In case if a cell has values, e.g., close to 0 or 1
                  val_avg /= values.size();
                  if (val_min < val_avg && val_avg < val_max)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();
                    }
                }

              if (cell->refine_flag_set())
                break;
            }
        }

      for (const auto &cell : tria.active_cell_iterators())
        {
          const auto cell_level = static_cast<unsigned int>(cell->level());

          if (cell->refine_flag_set() && cell_level == max_allowed_level)
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() && cell_level == min_allowed_level)
            cell->clear_coarsen_flag();

          // Coarsen cell if it is overrefined
          if (cell_level > max_allowed_level)
            {
              if (cell->refine_flag_set())
                cell->clear_refine_flag();
              cell->set_coarsen_flag();
            }
        }

      tria.prepare_coarsening_and_refinement();
    }
  } // namespace internal

  template <typename VectorType, typename Triangulation, int dim>
  void
  coarsen_and_refine_mesh(const VectorType &         solution_to_estimate,
                          Triangulation &            tria,
                          DoFHandler<dim> &          dof_handler,
                          const Quadrature<dim - 1> &quad,
                          const double               top_fraction_of_cells,
                          const double               bottom_fraction_of_cells,
                          const unsigned int         min_allowed_level,
                          const unsigned int         max_allowed_level,
                          const typename VectorType::value_type val_min = 0.05,
                          const typename VectorType::value_type val_max = 0.95)
  {
    // Mark cells and prepare refinement
    internal::prepare_coarsening_and_refinement(solution_to_estimate,
                                                tria,
                                                dof_handler,
                                                quad,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells,
                                                min_allowed_level,
                                                max_allowed_level,
                                                val_min,
                                                val_max);

    // Execute refinement
    tria.execute_coarsening_and_refinement();
  }

  template <typename VectorType, typename Triangulation, int dim>
  void
  coarsen_and_refine_mesh(
    VectorType &                                        solution,
    Triangulation &                                     tria,
    DoFHandler<dim> &                                   dof_handler,
    AffineConstraints<typename VectorType::value_type> &constraints,
    const Quadrature<dim - 1> &                         quad,
    const double                                        top_fraction_of_cells,
    const double                          bottom_fraction_of_cells,
    const unsigned int                    min_allowed_level,
    const unsigned int                    max_allowed_level,
    std::function<void(VectorType &)>     reinitializer,
    const typename VectorType::value_type val_min              = 0.05,
    const typename VectorType::value_type val_max              = 0.95,
    const unsigned int                    block_estimate_start = 0,
    const unsigned int block_estimate_end = numbers::invalid_unsigned_int)
  {
    // Build partitioner
    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());

    // Copy solution so that it has the right ghosting
    VectorType solution_copy(solution.n_blocks());
    for (unsigned int b = 0; b < solution_copy.n_blocks(); ++b)
      {
        solution_copy.block(b).reinit(partitioner);
        solution_copy.block(b).copy_locally_owned_data_from(solution.block(b));
      }

    for (unsigned int b = 0; b < solution_copy.n_blocks(); ++b)
      constraints.distribute(solution_copy.block(b));

    solution_copy.update_ghost_values();

    const auto solution_copy_estimate =
      solution_copy.create_view(block_estimate_start,
                                block_estimate_end !=
                                    numbers::invalid_unsigned_int ?
                                  block_estimate_end :
                                  solution_copy.n_blocks());

    // Mark cells and prepare refinement
    internal::prepare_coarsening_and_refinement(*solution_copy_estimate,
                                                tria,
                                                dof_handler,
                                                quad,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells,
                                                min_allowed_level,
                                                max_allowed_level,
                                                val_min,
                                                val_max);

    // Raw pointers
    std::vector<typename VectorType::BlockType *> solution_ptr(
      solution.n_blocks());
    std::vector<const typename VectorType::BlockType *> solution_copy_ptr(
      solution_copy.n_blocks());
    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
      {
        solution_ptr[b]      = &solution.block(b);
        solution_copy_ptr[b] = &solution_copy.block(b);
      }

    // Prepare solution transfer
    parallel::distributed::SolutionTransfer<dim, typename VectorType::BlockType>
      solution_trans(dof_handler);

    solution_trans.prepare_for_coarsening_and_refinement(solution_copy_ptr);

    // Execute refinement
    tria.execute_coarsening_and_refinement();

    // Reinitialize vector to be transfered
    reinitializer(solution);

    // Transfer solution
    solution_trans.interpolate(solution_ptr);

    // note: apply constraints since the Newton solver expects this
    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
      constraints.distribute(solution.block(b));
  }

  struct EnergyCoefficients
  {
    double A;
    double B;
    double kappa_c;
    double kappa_p;
  };

  EnergyCoefficients
  compute_energy_params(const double surface_energy,
                        const double gb_energy,
                        const double interface_width,
                        const double length_scale,
                        const double energy_scale)
  {
    const double scaled_gb_energy =
      gb_energy / energy_scale * length_scale * length_scale;

    const double scaled_surface_energy =
      surface_energy / energy_scale * length_scale * length_scale;

    const double kappa_c = 3.0 / 4.0 *
                           (2.0 * scaled_surface_energy - scaled_gb_energy) *
                           interface_width;
    const double kappa_p = 3.0 / 4.0 * scaled_gb_energy * interface_width;

    const double A =
      (12.0 * scaled_surface_energy - 7.0 * scaled_gb_energy) / interface_width;
    const double B = scaled_gb_energy / interface_width;

    EnergyCoefficients params{A, B, kappa_c, kappa_p};

    return params;
  }

} // namespace Sintering