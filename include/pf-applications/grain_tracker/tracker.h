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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <pf-applications/base/debug.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <functional>
#include <iterator>
#include <stack>
#include <type_traits>

#include "distributed_stitching.h"
#include "grain.h"
#include "mapper.h"
#include "output.h"
#include "periodicity_graph.h"
#include "remap_graph.h"
#include "remapping.h"
#include "segment.h"
#include "tracking.h"

namespace GrainTracker
{
  using namespace dealii;

  enum class GrainRepresentation
  {
    spherical,
    elliptical,
    wavefront
  };

  /* The grain tracker algo itself. */
  template <int dim, typename Number>
  class Tracker : public Mapper
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    Tracker(const DoFHandler<dim> &                 dof_handler,
            const parallel::TriangulationBase<dim> &tria,
            const bool                              greedy_init,
            const bool                              allow_new_grains,
            const bool                              fast_reassignment,
            const unsigned int                      max_order_parameters_num,
            const GrainRepresentation               grain_representation =
              GrainRepresentation::spherical,
            const double       threshold_lower         = 0.01,
            const double       threshold_new_grains    = 0.02,
            const double       buffer_distance_ratio   = 0.05,
            const double       buffer_distance_fixed   = 0.0,
            const unsigned int order_parameters_offset = 2,
            const bool         do_timing               = true,
            const bool         do_logging              = false,
            const bool         use_old_remap           = false)
      : dof_handler(dof_handler)
      , tria(tria)
      , greedy_init(greedy_init)
      , allow_new_grains(allow_new_grains)
      , fast_reassignment(fast_reassignment)
      , max_order_parameters_num(max_order_parameters_num)
      , grain_representation(grain_representation)
      , threshold_lower(threshold_lower)
      , threshold_new_grains(threshold_new_grains)
      , buffer_distance_ratio(buffer_distance_ratio)
      , buffer_distance_fixed(buffer_distance_fixed)
      , order_parameters_offset(order_parameters_offset)
      , do_logging(do_logging)
      , use_old_remap(use_old_remap)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(do_timing)
    {}

    std::shared_ptr<Tracker<dim, Number>>
    clone() const
    {
      MyScope scope(timer, "tracker::clone", timer.is_enabled());

      auto new_tracker =
        std::make_shared<Tracker<dim, Number>>(dof_handler,
                                               tria,
                                               greedy_init,
                                               allow_new_grains,
                                               fast_reassignment,
                                               max_order_parameters_num,
                                               grain_representation,
                                               threshold_lower,
                                               threshold_new_grains,
                                               buffer_distance_ratio,
                                               buffer_distance_fixed,
                                               order_parameters_offset,
                                               timer.is_enabled(),
                                               do_logging,
                                               use_old_remap);

      new_tracker->op_particle_ids           = this->op_particle_ids;
      new_tracker->particle_ids_to_grain_ids = this->particle_ids_to_grain_ids;
      new_tracker->grain_segment_ids_numbering =
        this->grain_segment_ids_numbering;
      new_tracker->n_total_segments        = this->n_total_segments;
      new_tracker->grains                  = this->grains;
      new_tracker->old_grains              = this->old_grains;
      new_tracker->active_order_parameters = this->active_order_parameters;

      return new_tracker;
    }

    /* Track grains over timesteps. The function returns a tuple of bool
     * variables which signify if any grains have been reassigned and if the
     * number of active order parameters has been changed.
     */
    std::tuple<unsigned int, bool, bool>
    track(const BlockVectorType &solution,
          const unsigned int     n_order_params,
          const bool             skip_reassignment = false)
    {
      ScopedName sc("tracker::track");
      MyScope    scope(timer, sc, timer.is_enabled());

      // Copy old grains
      old_grains = grains;
      grains.clear();

      // Now we do not assign grain indices when searching for grains
      const bool assign_indices = false;

      const auto new_grains =
        detect_grains(solution, n_order_params, assign_indices);

      // Numberer for new grains
      unsigned int grain_numberer =
        (!old_grains.empty()) ? (old_grains.rbegin()->first + 1) : 0;

      // Create map with the grains whose ids have been changed
      std::vector<std::vector<bool>> grains_ids_changed(
        particle_ids_to_grain_ids.size());

      for (unsigned int iop = 0; iop < particle_ids_to_grain_ids.size(); ++iop)
        grains_ids_changed[iop].assign(particle_ids_to_grain_ids[iop].size(),
                                       false);

      // Map of invalid grains if any detected.
      // Map is used here for compatibility with print_grains()
      std::map<unsigned int, Grain<dim>> invalid_grains;
      unsigned int                       numerator_invalid = 0;

      // We may output additional info for debug purposes
      std::map<unsigned int, unsigned int> new_grains_to_old;
      if (do_logging)
        {
          std::stringstream ss;
          new_grains_to_old =
            transfer_grain_ids(new_grains, old_grains, n_order_params, ss);

          pcout << ss.str();
        }
      else
        {
          new_grains_to_old =
            transfer_grain_ids(new_grains, old_grains, n_order_params);
        }

      // Create segments and transfer grain_ids for them. We also keep track of
      // the omitted grains to check at the end if we have forgotten anything.
      unsigned int n_skipped_grains = 0;
      for (const auto &[current_grain_id, new_grain] : new_grains)
        {
          unsigned int new_grain_id = new_grains_to_old.at(current_grain_id);

          // Dynamics of the new grain
          typename Grain<dim>::Dynamics new_dynamics = Grain<dim>::None;

          // Check the grain number
          if (new_grain_id == std::numeric_limits<unsigned int>::max())
            {
              if (new_grain.get_max_value() > threshold_new_grains)
                {
                  if (allow_new_grains)
                    {
                      // If new grains are allowed, then simply assign the next
                      // available index
                      new_grain_id = grain_numberer++;
                    }
                  else
                    {
                      // Otherwise add to the map of invalid grains
                      invalid_grains.emplace(
                        std::make_pair(numerator_invalid++, new_grain));
                    }
                }
              else
                {
                  ++n_skipped_grains;
                }
            }
          else
            {
              // clang-format off
              AssertThrow(old_grains.at(new_grain_id).get_order_parameter_id() 
                == new_grain.get_order_parameter_id(),
                ExcGrainsInconsistency(
                  std::string("Something got wrong with the order parameters numbering:\r\n") +
                  std::string("\r\n    new_grain_id = ") +
                  std::to_string(new_grain_id) + 
                  std::string("\r\n    old grain order parameter   = ") +
                  std::to_string(old_grains.at(new_grain_id).get_order_parameter_id()) + 
                  std::string("\r\n    new grain order parameter   = ") +
                  std::to_string(new_grain.get_order_parameter_id()) + 
                  std::string("This could have happened if track() or initial_setup()") +
                  std::string(" was invoked resulting in the grains reassignement but the") +
                  std::string(" subsequent remap() was not called for the solution vector(s).")
              ));
              // clang-format on

              // Check if the detected grain is growing or not
              const bool is_growing = new_grain.get_measure() >
                                      old_grains.at(new_grain_id).get_measure();
              new_dynamics =
                is_growing ? Grain<dim>::Growing : Grain<dim>::Shrinking;
            }

          // Insert new grain if it has been identified or new ones are allowed
          if (new_grain_id != std::numeric_limits<unsigned int>::max())
            {
              grains.emplace(std::make_pair(new_grain_id, new_grain));
              grains.at(new_grain_id).set_grain_id(new_grain_id);
              grains.at(new_grain_id).set_dynamics(new_dynamics);
            }

          // Update mapping if we changed the grain id
          if (new_grain_id != current_grain_id)
            {
              auto &particle_to_grain =
                particle_ids_to_grain_ids[new_grain.get_order_parameter_id()];

              for (unsigned int ip = 0; ip < particle_to_grain.size(); ip++)
                {
                  auto &pmap = particle_to_grain[ip];

                  if (grains_ids_changed[new_grain.get_order_parameter_id()]
                                        [ip] == false &&
                      pmap.first == current_grain_id)
                    {
                      pmap.first = new_grain_id;
                      grains_ids_changed[new_grain.get_order_parameter_id()]
                                        [ip] = true;
                    }
                }
            }
        }

      // Throw exception if some invalid grains have been detected
      if (!invalid_grains.empty())
        {
          std::ostringstream ss;
          ss << "Unable to match some new grains with old ones "
             << "from the previous configuration:" << std::endl;
          print_grains(invalid_grains, ss);

          // Thrown an exception
          AssertThrow(invalid_grains.empty(), ExcGrainsInconsistency(ss.str()));
        }

      // Check if all the new grains were mapped or identified as new
      AssertThrow(
        new_grains.size() == (grains.size() + n_skipped_grains),
        ExcGrainsInconsistency(
          std::string("Not all initially detected grains have been mapped") +
          std::string(" to the old ones or properly added as new:\r\n") +
          std::string("\r\n    # of detected grains = ") +
          std::to_string(new_grains.size()) +
          std::string("\r\n    # of mapped grains   = ") +
          std::to_string(grains.size())));

      // Variables to return the result
      bool         grains_reassigned = false;
      bool         op_number_changed = false;
      unsigned int n_collisions      = 0;

      if (skip_reassignment == false)
        {
          /* For tracking we want the grains assigned to the same order
           * parameter to be as far from each other as possible to reduce the
           * number of costly grains reassignment.
           */
          const bool force_reassignment = false;

          // Reassign grains
          std::tie(n_collisions, grains_reassigned) =
            reassign_grains(force_reassignment, fast_reassignment);

          // Check if number of order parameters has changed
          op_number_changed =
            (active_order_parameters.size() != n_order_params);
        }
      else
        {
          // Build active order parameters
          active_order_parameters = extract_active_order_parameter_ids(grains);
        }

      // Build inverse mapping after all indices are set
      build_inverse_mapping();

      return std::make_tuple(n_collisions,
                             grains_reassigned,
                             op_number_changed);
    }

    /* Initialization of grains at the very first step. The function returns a
     * tuple of bool variables which signify if any grains have been reassigned
     * and if the number of active order parameters has been changed.
     */
    std::tuple<unsigned int, bool, bool>
    initial_setup(const BlockVectorType &solution,
                  const unsigned int     n_order_params,
                  const bool             skip_reassignment = false)
    {
      ScopedName sc("tracker::initial_setup");
      MyScope    scope(timer, sc, timer.is_enabled());

      const bool assign_indices = true;

      grains = detect_grains(solution, n_order_params, assign_indices);

      bool         grains_reassigned = false;
      bool         op_number_changed = false;
      unsigned int n_collisions      = 0;

      if (skip_reassignment == false)
        {
          /* Initial grains reassignment, the closest neighbors are allowed as
           * we want to minimize the number of order parameters in use.
           */
          const bool force_reassignment = greedy_init;

          // Reassign grains
          std::tie(n_collisions, grains_reassigned) =
            reassign_grains(force_reassignment, fast_reassignment);

          // Check if number of order parameters has changed
          op_number_changed =
            (active_order_parameters.size() != n_order_params);
        }
      else
        {
          // Build active order parameters
          active_order_parameters = extract_active_order_parameter_ids(grains);
        }

      // Build inverse mapping after grains are detected
      build_inverse_mapping();

      return std::make_tuple(n_collisions,
                             grains_reassigned,
                             op_number_changed);
    }

    // Remap a single state vector
    unsigned int
    remap(BlockVectorType &solution) const
    {
      return remap({&solution});
    }

    // Remap state vectors
    unsigned int
    remap(std::vector<std::shared_ptr<BlockVectorType>> solutions) const
    {
      std::vector<BlockVectorType *> raw_ptrs;

      std::transform(solutions.begin(),
                     solutions.end(),
                     std::back_inserter(raw_ptrs),
                     [](auto &sol) { return sol.get(); });

      return remap(raw_ptrs);
    }

    unsigned int
    remap(std::vector<BlockVectorType *> solutions) const
    {
      return use_old_remap ? remap_old(solutions) : remap_new(solutions);
    }

  private:
    // Remap state vectors
    unsigned int
    remap_new(std::vector<BlockVectorType *> solutions) const
    {
      ScopedName sc("tracker::remap");
      MyScope    scope(timer, sc, timer.is_enabled());

      // Vector for dof values transfer
      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      // Build a list of remappings for grains
      std::map<unsigned int, Remapping> grain_id_to_remapping;
      for (const auto &[gid, grain] : grains)
        {
          (void)gid;

          if (grain.get_order_parameter_id() !=
              grain.get_old_order_parameter_id())
            {
              grain_id_to_remapping.emplace(
                grain.get_grain_id(),
                Remapping(grain.get_grain_id(),
                          grain.get_old_order_parameter_id(),
                          grain.get_order_parameter_id()));
            }
        }

      // Total number of grains remapped
      unsigned int n_grains_remapped = grain_id_to_remapping.size();

      if (n_grains_remapped == 0)
        return n_grains_remapped;

      // Init remapping cache
      std::map<std::vector<unsigned int>, std::list<Remapping>>
        remappings_cache;

      // Copy solutions
      std::vector<std::shared_ptr<BlockVectorType>> solutions_copy;
      for (const auto solution : solutions)
        solutions_copy.push_back(std::make_shared<BlockVectorType>(*solution));

      // Main remapping loop
      for (auto &cell : dof_handler.active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          const auto cell_index = cell->global_active_cell_index();

          std::vector<unsigned int> grains_at_cell;

          for (unsigned int op = 0; op < op_particle_ids.n_blocks(); ++op)
            {
              const auto particle_id_for_op =
                get_particle_index(op, cell_index);

              if (particle_id_for_op != numbers::invalid_unsigned_int)
                {
                  const auto grain_id =
                    get_grain_and_segment(op, particle_id_for_op).first;

                  grains_at_cell.push_back(grain_id);
                }
            }

          // If no any grain at the current cell then skip the rest
          if (grains_at_cell.empty())
            continue;

          // Try to get the cached value
          auto it_cache = remappings_cache.find(grains_at_cell);

          // Build the remappings sequence if nothing in cache
          if (it_cache == remappings_cache.end())
            {
              // Build local remappings at the cell
              std::list<Remapping> remappings;
              for (const auto &gid : grains_at_cell)
                {
                  const auto it_remap = grain_id_to_remapping.find(gid);

                  if (it_remap != grain_id_to_remapping.end())
                    remappings.insert(remappings.end(), it_remap->second);
                }

              // The local remappings now have to checked for collisions
              // Build graph to resolve overlapping remappings
              RemapGraph graph;

              // Check for collisions in the remappings
              for (const auto &ri : remappings)
                {
                  // const auto &grain_i = grains.at(ri.grain_id);

                  for (const auto &rj : remappings)
                    {
                      // const auto &grain_j = grains.at(rj.grain_id);

                      if (ri != rj)
                        {
                          /*
                          const double buffer_i = grain_i.transfer_buffer();
                          const double buffer_j = grain_j.transfer_buffer();

                          const bool has_overlap =
                            grain_i.distance(grain_j) - buffer_i - buffer_j < 0;
                          */

                          // TODO: verify that the check below can be removed
                          constexpr bool has_overlap = true;

                          /* If the two grains involved in remappings overlap
                           * and share the same order parameter in the current
                           * and previous states, then we add them for analysis
                           * to the graph.
                           */
                          if (has_overlap && ri.to == rj.from)
                            {
                              graph.add_remapping(ri.from, ri.to, ri.grain_id);

                              /* Besides that, we need to add also the
                               * subsequent remapping for the second grain to
                               * the graph too.
                               */
                              auto it_re = std::find_if(
                                remappings.begin(),
                                remappings.end(),
                                [target_grain_id = rj.grain_id](const auto &a) {
                                  return a.grain_id == target_grain_id;
                                });

                              AssertThrow(it_re != remappings.end(),
                                          ExcMessage(
                                            "Particles collision detected!"));

                              graph.add_remapping(it_re->from,
                                                  it_re->to,
                                                  it_re->grain_id);
                            }
                        }
                    }
                }

              /* If graph is not empty, then have some dependencies in remapping
               * and need to perform at first those at the end of the graph in
               * order not to break the configuration of the domain.
               */
              if (!graph.empty())
                {
                  // At frist resolve cyclic remappings if any cycle exists
                  const auto remappings_via_temp =
                    graph.resolve_cycles(remappings);

                  // Then rearrange the rest
                  graph.rearrange(remappings);

                  for (const auto &[to_temp, from_temp] : remappings_via_temp)
                    {
                      remappings.insert(remappings.begin(), to_temp);
                      remappings.insert(remappings.end(), from_temp);
                    }
                }

              // Update iterator to the recently added remapping sequence
              bool status = false;
              std::tie(it_cache, status) =
                remappings_cache.emplace(grains_at_cell, remappings);

              AssertThrow(
                status,
                ExcMessage(
                  "Failed to insert remappings into cache for cells with grains " +
                  debug::to_string(grains_at_cell)));
            }
          const auto &remap_sequence = it_cache->second;

          for (unsigned int i = 0; i < solutions.size(); ++i)
            {
              const auto &solution      = solutions[i];
              const auto &solution_copy = solutions_copy[i];

              // Prepare temp storage
              std::stack<Vector<Number>> temp_values;

              for (const auto &re : remap_sequence)
                {
                  // Retrieve values from the existing block or temp storage
                  if (re.from != numbers::invalid_unsigned_int)
                    {
                      const unsigned int op_id_src =
                        re.from + order_parameters_offset;
                      cell->get_dof_values(solution_copy->block(op_id_src),
                                           values);
                    }
                  else
                    {
                      values = temp_values.top();
                      temp_values.pop();
                    }

                  // Set values to the existing block or temp storage
                  if (re.to != numbers::invalid_unsigned_int)
                    {
                      const unsigned int op_id_dst =
                        re.to + order_parameters_offset;
                      cell->set_dof_values(values, solution->block(op_id_dst));
                    }
                  else
                    {
                      temp_values.push(values);
                    }

                  // Nullify source
                  if (re.from != numbers::invalid_unsigned_int)
                    {
                      const unsigned int op_id_src =
                        re.from + order_parameters_offset;
                      values = 0;
                      cell->set_dof_values(values, solution->block(op_id_src));
                    }
                }
            }
        }

      return n_grains_remapped;
    }

    unsigned int
    remap_old(std::vector<BlockVectorType *> solutions) const
    {
      ScopedName sc("tracker::remap");
      MyScope    scope(timer, sc, timer.is_enabled());

      // Logging for remapping
      std::vector<std::string> log;

      // Vector for dof values transfer
      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      // lambda to perform certain modifications of the solution dof values
      // clang-format off
      auto alter_dof_values_for_grain =
        [this, &solutions] (const Grain<dim> &grain,
          std::function<void(
            const dealii::DoFCellAccessor<dim, dim, false> &cell,
            BlockVectorType *solution)> callback) {
          // clang-format on

          ScopedName sc("alter_dof_values_for_grain");
          MyScope    scope(timer, sc, timer.is_enabled());

          const double transfer_buffer = grain.transfer_buffer();

          for (auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  bool in_grain = false;
                  for (const auto &segment : grain.get_segments())
                    {
                      if (cell->barycenter().distance(segment.get_center()) <
                          segment.get_radius() + transfer_buffer)
                        {
                          in_grain = true;
                          break;
                        }
                    }

                  if (in_grain)
                    {
                      for (auto &solution : solutions)
                        {
                          callback(*cell, solution);
                        }
                    }
                }
            }
        };

      // At first let us clean up those grains which disappered completely
      std::map<unsigned int, Grain<dim>> disappered_grains;
      std::set_difference(
        old_grains.begin(),
        old_grains.end(),
        grains.begin(),
        grains.end(),
        std::inserter(disappered_grains, disappered_grains.end()),
        [](const auto &a, const auto &b) { return a.first < b.first; });

      // We will fill the blocks with zeros
      values = 0;
      for (const auto &[gid, gr] : disappered_grains)
        {
          const unsigned int op_id =
            gr.get_order_parameter_id() + order_parameters_offset;

          if (do_logging)
            {
              std::ostringstream ss;
              ss << "Grain " << gr.get_grain_id() << " having order parameter "
                 << op_id << " has disappered" << std::endl;
              log.emplace_back(ss.str());
            }

          alter_dof_values_for_grain(
            gr,
            [this,
             &values,
             op_id](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                    BlockVectorType *                               solution) {
              cell.set_dof_values(values, solution->block(op_id));
            });
        }

      // Build a sequence of remappings
      std::list<Remapping> remappings;
      for (const auto &[gid, grain] : grains)
        {
          (void)gid;

          if (grain.get_order_parameter_id() !=
              grain.get_old_order_parameter_id())
            {
              remappings.emplace(remappings.end(),
                                 grain.get_grain_id(),
                                 grain.get_old_order_parameter_id(),
                                 grain.get_order_parameter_id());
            }
        }

      // Total number of grains remapped
      unsigned int n_grains_remapped = remappings.size();

      // Build graph to resolve overlapping remappings
      RemapGraph graph;

      // Check for collisions in the remappings
      for (const auto &ri : remappings)
        {
          const auto &grain_i = grains.at(ri.grain_id);

          for (const auto &rj : remappings)
            {
              const auto &grain_j = grains.at(rj.grain_id);

              if (ri != rj)
                {
                  const double buffer_i = grain_i.transfer_buffer();
                  const double buffer_j = grain_j.transfer_buffer();

                  const bool has_overlap =
                    grain_i.distance(grain_j) - buffer_i - buffer_j < 0;

                  /* If the two grains involved in remappings overlap and share
                   * the same order parameter in the current and previous
                   * states, then we add them for analysis to the graph.
                   */
                  if (has_overlap && ri.to == rj.from)
                    {
                      graph.add_remapping(ri.from, ri.to, ri.grain_id);

                      /* Besides that, we need to add also the subsequent
                       * remapping for the second grain to the graph too.
                       */

                      auto it_re =
                        std::find_if(remappings.begin(),
                                     remappings.end(),
                                     [target_grain_id =
                                        rj.grain_id](const auto &a) {
                                       return a.grain_id == target_grain_id;
                                     });

                      AssertThrow(it_re != remappings.end(),
                                  ExcMessage("Particles collision detected!"));

                      graph.add_remapping(it_re->from,
                                          it_re->to,
                                          it_re->grain_id);
                    }
                }
            }
        }

      // Transfer cycled grains to temporary vectors
      std::vector<std::pair<Remapping, Remapping>> remappings_via_temp;

      AssertThrowDistributedDimension(
        (static_cast<unsigned int>(graph.empty())));

      /* If graph is not empty, then have some dependencies in remapping and
       * need to perform at first those at the end of the graph in order not to
       * break the configuration of the domain.
       */
      if (!graph.empty())
        {
          ScopedName sc("resolve_cycles");
          MyScope    scope(timer, sc, timer.is_enabled());

          /* Check if the graph has cycles - these are unlikely situations and
           * at the moment we do not handle them due to complexity.
           */

          if (do_logging)
            {
              std::ostringstream ss;
              ss << "Remapping dependencies have been detected and resolved."
                 << std::endl;
              graph.print(ss);
              log.emplace_back(ss.str());
            }

          AssertThrowDistributedDimension(remappings.size());

          // At frist resolve cyclic remappings
          remappings_via_temp = graph.resolve_cycles(remappings);

          // Then rearrange the rest
          graph.rearrange(remappings);
        }

      // Create temporary vectors for grain transfers
      std::map<const BlockVectorType *, std::shared_ptr<BlockVectorType>>
        solutions_to_temps;

      if (!remappings_via_temp.empty())
        {
          const auto partitioner =
            std::make_shared<Utilities::MPI::Partitioner>(
              dof_handler.locally_owned_dofs(),
              DoFTools::extract_locally_relevant_dofs(dof_handler),
              dof_handler.get_communicator());

          AssertThrowDistributedDimension(solutions.size());

          for (const auto &solution : solutions)
            {
              /* Sicne boost graphs algorithms are not deterministic, the number
               * of remapping performed with the aid of temporary vectors may
               * vary. We then create a temporary block vector of the maximum
               * size to fit all ranks. However, we may also think of picking up
               * a remapping sequence among all of the available ones
               * with thee smallest number of remapping steps. */
              const auto max_size =
                Utilities::MPI::max(remappings_via_temp.size(), MPI_COMM_WORLD);

              auto temp = std::make_shared<BlockVectorType>(max_size);
              for (unsigned int b = 0; b < temp->n_blocks(); ++b)
                {
                  temp->block(b).reinit(partitioner);
                  temp->block(b).update_ghost_values();
                }

              solutions_to_temps.emplace(solution, temp);
            }
        }

      // Transfer some grains to temp vectors to break the cycles
      for (auto it = remappings_via_temp.cbegin();
           it != remappings_via_temp.cend();
           ++it)
        {
          const auto &re    = it->first;
          const auto &grain = grains.at(re.grain_id);

          if (do_logging)
            {
              std::ostringstream ss;
              ss << "Remap order parameter for grain id = " << re.grain_id
                 << ": from " << re.from << " to temp" << std::endl;
              log.emplace_back(ss.str());
            }

          const unsigned int op_id_src = re.from + order_parameters_offset;
          const unsigned int op_id_dst = it - remappings_via_temp.begin();

          /* At first we transfer the values from the dofs related to the
           * old order parameters to the temporary blocks.
           */
          alter_dof_values_for_grain(
            grain,
            [this, &values, op_id_src, op_id_dst, &solutions_to_temps](
              const dealii::DoFCellAccessor<dim, dim, false> &cell,
              BlockVectorType *                               solution) {
              cell.get_dof_values(solution->block(op_id_src), values);
              cell.set_dof_values(
                values, solutions_to_temps.at(solution)->block(op_id_dst));
            });

          // Then we iterate again to nullify the old dofs
          values = 0;
          alter_dof_values_for_grain(
            grain,
            [this,
             &values,
             op_id_src](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                        BlockVectorType *solution) {
              cell.set_dof_values(values, solution->block(op_id_src));
            });
        }

      // Now transfer values for the remaining grains
      for (const auto &re : remappings)
        {
          const auto &grain = grains.at(re.grain_id);

          /* Transfer buffer is the extra zone around the grain within which
           * the order parameters are swapped. Its maximum size is the half
           * of the distance to the nearest neighbor.
           */
          const unsigned int op_id_src = re.from + order_parameters_offset;
          const unsigned int op_id_dst = re.to + order_parameters_offset;

          if (do_logging)
            {
              std::ostringstream ss;
              ss << "Remap order parameter for grain id = " << re.grain_id
                 << ": from " << re.from << " to " << re.to << std::endl;
              log.emplace_back(ss.str());
            }

          /* At first we transfer the values from the dofs related to the
           * old order parameters to the dofs of the new order parameter.
           */
          alter_dof_values_for_grain(
            grain,
            [this, &values, op_id_src, op_id_dst](
              const dealii::DoFCellAccessor<dim, dim, false> &cell,
              BlockVectorType *                               solution) {
              cell.get_dof_values(solution->block(op_id_src), values);
              cell.set_dof_values(values, solution->block(op_id_dst));
            });

          // Then we iterate again to nullify the old dofs
          values = 0;
          alter_dof_values_for_grain(
            grain,
            [this,
             &values,
             op_id_src](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                        BlockVectorType *solution) {
              cell.set_dof_values(values, solution->block(op_id_src));
            });
        }

      // Transfer the grains from temp to where they had to be
      for (auto it = remappings_via_temp.cbegin();
           it != remappings_via_temp.cend();
           ++it)
        {
          const auto &re    = it->second;
          const auto &grain = grains.at(re.grain_id);

          if (do_logging)
            {
              std::ostringstream ss;
              ss << "Remap order parameter for grain id = " << re.grain_id
                 << ": from temp to " << re.to << std::endl;
              log.emplace_back(ss.str());
            }

          const unsigned int op_id_src = it - remappings_via_temp.begin();
          const unsigned int op_id_dst = re.to + order_parameters_offset;

          /* At first we transfer the values from the dofs related to the
           * old order parameters to the temporary blocks.
           */
          alter_dof_values_for_grain(
            grain,
            [this, &values, op_id_src, op_id_dst, &solutions_to_temps](
              const dealii::DoFCellAccessor<dim, dim, false> &cell,
              BlockVectorType *                               solution) {
              cell.get_dof_values(
                solutions_to_temps.at(solution)->block(op_id_src), values);
              cell.set_dof_values(values, solution->block(op_id_dst));
            });

          /* We do not need to iterate again to nullify the old dofs since the
           * temporary vectors will get deleted
           */
        }

      if (do_logging)
        print_log(log);

      return n_grains_remapped;
    }

  public:
    const std::map<unsigned int, Grain<dim>> &
    get_grains() const
    {
      return grains;
    }

    template <typename InputIt,
              typename = std::enable_if_t<std::is_same_v<
                typename std::iterator_traits<InputIt>::value_type,
                Grain<dim>>>>
    void
    load_grains(InputIt first, InputIt last)
    {
      for (; first != last; ++first)
        grains.emplace(first->get_grain_id(), *first);
    }

    template <typename OutputIt>
    void
    save_grains(OutputIt output) const
    {
      for (const auto &[gid, grain] : grains)
        *output++ = grain;
    }

    // Get active order parameters ids
    std::set<unsigned int>
    get_active_order_parameters() const
    {
      return active_order_parameters;
    }

    // Print last grains
    template <typename Stream>
    void
    print_current_grains(Stream &out, bool invariant = false) const
    {
      if (invariant)
        print_grains_invariant(grains, out);
      else
        print_grains(grains, out);
    }

    // Print last grains
    template <typename Stream>
    void
    print_old_grains(Stream &out, bool invariant = false) const
    {
      if (invariant)
        print_grains_invariant(old_grains, out);
      else
        print_grains(old_grains, out);
    }

    // Output last grains
    void
    output_current_grains(std::string prefix = std::string("grains")) const
    {
      output_grains(grains, prefix);
    }

    /* If there is any particle located at a given cell within a given order
     * parameter, then index is returned, otherwise -
     * numbers::invalid_unsigned_int. */
    unsigned int
    get_particle_index(const unsigned int order_parameter,
                       const unsigned int cell_index) const override
    {
      AssertThrow(order_parameter < op_particle_ids.n_blocks(),
                  ExcMessage("Invalid order_parameter id = " +
                             std::to_string(order_parameter) +
                             " provided, total number of particles = " +
                             std::to_string(op_particle_ids.n_blocks())));
      const auto &particle_ids = op_particle_ids.block(order_parameter);

      AssertThrow(cell_index < particle_ids.size(),
                  ExcMessage(
                    "Invalid cell_index = " + std::to_string(cell_index) +
                    " provided, total number of cells = " +
                    std::to_string(particle_ids.size())));
      const auto &particle_id = particle_ids[cell_index];

      return (particle_id == invalid_particle_id) ?
               numbers::invalid_unsigned_int :
               static_cast<unsigned int>(particle_id);
    }

    /* If there is a grain attached to particle_id, then the pair of grain_id
     * and segment_id is returned, otherwise - numbers::invalid_unsigned_int.
     * Note, that despite a cell may contain valid particle_id, the
     * corresponding grain_id still may be equal to
     * numbers::invalid_unsigned_int. This can happen if an isolated area, whose
     * maximum value is larger than threshold_lower, has been detected but that
     * value is smaller than threshold_new_grains. This means that the emerging
     * isolated area is not yet large enough to be promoted to a independent
     * grain, it may turn out to be a spurious area. If such an area is
     * detected, it obtains a valid particle_id when track() is called, but then
     * later it gets filtered out when detect_grains() is invoked and its
     * grain_id is set to numbers::invalid_unsigned_int. */
    std::pair<unsigned int, unsigned int>
    get_grain_and_segment(const unsigned int order_parameter,
                          const unsigned int particle_id) const override
    {
      AssertThrow(particle_id != static_cast<unsigned int>(invalid_particle_id),
                  ExcMessage("Invalid particle_id provided"));

      return particle_ids_to_grain_ids[order_parameter][particle_id];
    }

    const Point<dim> &
    get_segment_center(const unsigned int grain_id,
                       const unsigned int segment_id) const
    {
      Assert(grains.find(grain_id) != grains.cend(),
             ExcMessage("Grain with grain_id = " + std::to_string(grain_id) +
                        " does not exist in the grains map"));

      return grains.at(grain_id).get_segments()[segment_id].get_center();
    }

    unsigned int
    get_grain_segment_index(const unsigned int grain_id,
                            const unsigned int segment_id) const override
    {
      Assert(grain_segment_ids_numbering.find(grain_id) !=
               grain_segment_ids_numbering.cend(),
             ExcMessage("Grain with grain_id = " + std::to_string(grain_id) +
                        " does not exist in the inverse mapping"));

      const auto &grain = grain_segment_ids_numbering.at(grain_id);

      Assert(grain.find(segment_id) != grain.cend(),
             ExcMessage(
               "Segment with segment_id = " + std::to_string(segment_id) +
               " does not exist in the grain with grain_id = " +
               std::to_string(grain_id)));

      return grain.at(segment_id);
    }

    unsigned int
    n_segments() const override
    {
      return n_total_segments;
    }

    bool
    empty() const override
    {
      return grains.empty();
    }

    void
    custom_reassignment(
      std::function<void(std::map<unsigned int, Grain<dim>> &)> callback)
    {
      callback(grains);

      // Rebuild completely active order parameters
      active_order_parameters = extract_active_order_parameter_ids(grains);
    }

  private:
    std::map<unsigned int, Grain<dim>>
    detect_grains(const BlockVectorType &solution,
                  const unsigned int     n_order_params,
                  const bool             assign_indices)
    {
      ScopedName sc("detect_grains");
      MyScope    scope(timer, sc, timer.is_enabled());

      std::map<unsigned int, Grain<dim>> new_grains;

      const MPI_Comm comm = MPI_COMM_WORLD;

      constexpr bool use_stitching_via_graphs = true;

      // Numerator
      unsigned int grains_numerator = 0;

      // Order parameter indices stored per cell
      op_particle_ids.reinit(n_order_params);
      for (unsigned int b = 0; b < op_particle_ids.n_blocks(); ++b)
        op_particle_ids.block(b).reinit(
          tria.global_active_cell_index_partitioner().lock());

      particle_ids_to_grain_ids.clear();
      particle_ids_to_grain_ids.resize(n_order_params);

      // Cache particle max values
      std::vector<std::vector<double>> op_particle_max_values(n_order_params);

      for (unsigned int current_order_parameter_id = 0;
           current_order_parameter_id < n_order_params;
           ++current_order_parameter_id)
        {
          auto &particle_ids =
            op_particle_ids.block(current_order_parameter_id);

          const auto &solution_order_parameter = solution.block(
            current_order_parameter_id + order_parameters_offset);

          // Run flooding, build local particle ids and execute stitching
          const auto [offset,
                      local_to_global_particle_ids,
                      local_particle_max_values] =
            detect_local_particle_groups(particle_ids,
                                         dof_handler,
                                         solution_order_parameter,
                                         use_stitching_via_graphs,
                                         threshold_lower,
                                         invalid_particle_id,
                                         timer.is_enabled() ? &timer : nullptr);

          // Get particle max values
          op_particle_max_values[current_order_parameter_id] =
            compute_particles_max_values(dof_handler,
                                         particle_ids,
                                         local_to_global_particle_ids,
                                         offset,
                                         invalid_particle_id,
                                         local_particle_max_values);

          // Set global ids to the particles
          switch_to_global_indices(particle_ids,
                                   local_to_global_particle_ids,
                                   offset,
                                   invalid_particle_id);
        }

      // Get direct neighbors if the wavefront representation is used
      std::map<std::pair<unsigned int, unsigned int>,
               std::set<std::pair<unsigned int, unsigned int>>>
        direct_neighbors;
      if (grain_representation == GrainRepresentation::wavefront)
        direct_neighbors = get_direct_neighbors(dof_handler,
                                                op_particle_ids,
                                                invalid_particle_id);

      for (unsigned int current_order_parameter_id = 0;
           current_order_parameter_id < n_order_params;
           ++current_order_parameter_id)
        {
          const auto &particle_ids =
            op_particle_ids.block(current_order_parameter_id);

          const auto &particle_max_values =
            op_particle_max_values[current_order_parameter_id];

          const unsigned int n_particles = particle_max_values.size();

          // Determine properties of particles (volume, radius, center, etc)
          const auto [particle_centers, particle_measures] =
            compute_particles_info(dof_handler,
                                   particle_ids,
                                   n_particles,
                                   invalid_particle_id);

          // Compute particles radii and remote points (if needed)
          const auto [particle_radii, particle_remotes] =
            compute_particles_radii(dof_handler,
                                    particle_ids,
                                    particle_centers,
                                    grain_representation ==
                                      GrainRepresentation::elliptical,
                                    invalid_particle_id);

          // Compute particles inertia if needed
          std::vector<double> particle_inertia;
          if (grain_representation == GrainRepresentation::elliptical)
            particle_inertia = compute_particles_inertia(dof_handler,
                                                         particle_ids,
                                                         particle_centers,
                                                         invalid_particle_id);


          // Build periodicity between particles
          std::set<std::tuple<unsigned int, unsigned int>> periodicity;

          for (const auto &cell :
               dof_handler.get_triangulation().active_cell_iterators())
            if (!cell->is_artificial())
              {
                const auto particle_id =
                  particle_ids[cell->global_active_cell_index()];

                if (particle_id == invalid_particle_id)
                  continue;

                for (const auto face : cell->face_indices())
                  {
                    if (!cell->has_periodic_neighbor(face))
                      continue;

                    const auto add = [&](const auto &other_cell) {
                      if (other_cell->is_locally_owned() == false)
                        return;

                      const auto neighbor_particle_id =
                        particle_ids[other_cell->global_active_cell_index()];

                      if (neighbor_particle_id == invalid_particle_id)
                        return;

                      periodicity.emplace(neighbor_particle_id, particle_id);
                    };

                    if (cell->periodic_neighbor(face)->has_children())
                      {
                        for (unsigned int subface = 0;
                             subface < GeometryInfo<dim>::n_subfaces(
                                         dealii::internal::SubfaceCase<
                                           dim>::case_isotropic);
                             ++subface)
                          add(
                            cell->periodic_neighbor_child_on_subface(face,
                                                                     subface));
                      }
                    else
                      add(cell->periodic_neighbor(face));
                  }
              }

          // Convert set to flatten vector
          std::vector<unsigned int> periodicity_flatten;
          for (const auto &conn : periodicity)
            {
              periodicity_flatten.push_back(std::get<0>(conn));
              periodicity_flatten.push_back(std::get<1>(conn));
            }

          // Perform global communication, the data is not large
          auto global_periodicity =
            Utilities::MPI::all_gather(comm, periodicity_flatten);

          // Build periodicity graph
          PeriodicityGraph pg;
          for (const auto &part_periodicity : global_periodicity)
            for (unsigned int i = 0; i < part_periodicity.size(); i += 2)
              pg.add_connection(part_periodicity[i], part_periodicity[i + 1]);

          // Build particles groups
          std::vector<unsigned int> particle_groups(
            n_particles, numbers::invalid_unsigned_int);

          const unsigned int n_groups_found = pg.build_groups(particle_groups);

          // Indices of free particles (all at the beginning)
          std::set<unsigned int> free_particles;
          for (unsigned int i = 0; i < n_particles; i++)
            free_particles.insert(i);

          // Initialize the mapping vector
          particle_ids_to_grain_ids[current_order_parameter_id].resize(
            n_particles);

          // Build particle distances data if wavefront representation is used
          std::vector<std::map<std::pair<unsigned int, unsigned int>, double>>
            particle_distances(n_particles);
          if (grain_representation == GrainRepresentation::wavefront)
            {
              // Estimate particle distances
              const auto distances_estimation =
                estimate_particle_distances(particle_ids,
                                            dof_handler,
                                            invalid_particle_id,
                                            timer.is_enabled() ? &timer :
                                                                 nullptr);

              // Build particle distances data
              particle_distances.resize(n_particles);

              for (const auto &[key, dist] : distances_estimation)
                {
                  particle_distances[key.first].emplace(
                    std::make_pair(current_order_parameter_id, key.second),
                    dist);
                  particle_distances[key.second].emplace(
                    std::make_pair(current_order_parameter_id, key.first),
                    dist);
                }
              for (const auto &[primary, secondaries] : direct_neighbors)
                if (primary.first == current_order_parameter_id)
                  for (const auto &secondary : secondaries)
                    particle_distances[primary.second].emplace(
                      std::make_pair(secondary.first, secondary.second), 0);
            }

          // Lambda to create grains and segments
          auto append_segment = [&,
                                 grain_representation = grain_representation](
                                  unsigned int index, unsigned int grain_id) {
            new_grains.try_emplace(grain_id,
                                   assign_indices ?
                                     grain_id :
                                     numbers::invalid_unsigned_int,
                                   current_order_parameter_id);

            std::unique_ptr<Representation> representation;

            if (grain_representation == GrainRepresentation::spherical)
              {
                representation = std::make_unique<RepresentationSpherical<dim>>(
                  particle_centers[index], particle_radii[index]);
              }
            else if (grain_representation == GrainRepresentation::elliptical)
              {
                auto representation_el =
                  std::make_unique<RepresentationElliptical<dim>>(
                    particle_centers[index],
                    particle_measures[index],
                    &(particle_inertia[index * num_inertias<dim>]));

                const auto remote_point =
                  particle_centers[index] + particle_remotes[index];
                const auto &E = representation_el->ellipsoid;
                const auto [t_inter, overlap] =
                  find_ellipsoid_intersection(E, E.get_center(), remote_point);

                // This means that we need to enlarge the radii
                if (t_inter > 0 && t_inter < 1)
                  {
                    const auto scale = 1. + (1 - t_inter);

                    const auto &center =
                      representation_el->ellipsoid.get_center();
                    const auto &radii =
                      representation_el->ellipsoid.get_radii();
                    const auto &axes = representation_el->ellipsoid.get_axes();

                    std::array<double, dim> radii_scaled;
                    std::transform(radii.cbegin(),
                                   radii.cend(),
                                   radii_scaled.begin(),
                                   [&scale](double r) { return r * scale; });

                    // Recreate representation
                    representation =
                      std::make_unique<RepresentationElliptical<dim>>(
                        center, radii_scaled, axes);
                  }
                else
                  {
                    representation = std::move(representation_el);
                  }
              }
            else if (grain_representation == GrainRepresentation::wavefront)
              {
                representation = std::make_unique<RepresentationWavefront<dim>>(
                  current_order_parameter_id,
                  index,
                  particle_centers[index],
                  particle_distances[index]);
              }
            else
              AssertThrow(false, ExcNotImplemented());

            new_grains.at(grain_id).add_segment(
              Segment<dim>(particle_centers[index],
                           particle_radii[index],
                           particle_measures[index],
                           particle_max_values[index],
                           std::move(representation)));

            const unsigned int last_segment_id =
              new_grains.at(grain_id).n_segments() - 1;

            particle_ids_to_grain_ids[current_order_parameter_id][index] =
              std::make_pair(grain_id, last_segment_id);
          };

          // Parse groups at first to create grains
          for (unsigned int i = 0; i < n_particles; ++i)
            {
              if (particle_groups[i] != numbers::invalid_unsigned_int)
                {
                  unsigned int grain_id = particle_groups[i] + grains_numerator;

                  append_segment(i, grain_id);

                  free_particles.erase(i);
                }
            }

          grains_numerator += n_groups_found;

          // Then handle the remaining non-periodic particles
          for (const unsigned int i : free_particles)
            {
              unsigned int grain_id = grains_numerator;

              append_segment(i, grain_id);

              ++grains_numerator;
            }
        }

      AssertThrow(
        grains_numerator == static_cast<unsigned int>(new_grains.size()),
        ExcMessage(
          "Inconsistent grains numbering: grains_numerator = " +
          std::to_string(grains_numerator) +
          " and new_grains.size() = " + std::to_string(new_grains.size())));

      return new_grains;
    }

    // Build a set of active order parameters
    std::set<unsigned int>
    extract_active_order_parameter_ids(
      const std::map<unsigned int, Grain<dim>> &all_grains) const
    {
      std::set<unsigned int> active_op_ids;

      for (const auto &[gid, gr] : all_grains)
        {
          (void)gid;
          active_op_ids.insert(gr.get_order_parameter_id());
        }

      return active_op_ids;
    }

    // Reassign grains order parameters to prevent collision
    std::tuple<unsigned int, bool>
    reassign_grains(const bool force_reassignment,
                    const bool try_fast_reassignment)
    {
      ScopedName sc("reassign_grains");
      MyScope    scope(timer, sc, timer.is_enabled());

      bool grains_reassigned = false;

      std::vector<std::string> log;

      // DSP for colorization if order parameters are compressed
      const unsigned int                   n_grains = grains.size();
      DynamicSparsityPattern               dsp(n_grains);
      std::map<unsigned int, unsigned int> grains_to_sparsity;

      unsigned int id_counter = 0;
      std::transform(grains.begin(),
                     grains.end(),
                     std::inserter(grains_to_sparsity,
                                   grains_to_sparsity.end()),
                     [&id_counter](const auto &a) {
                       return std::make_pair(a.first, id_counter++);
                     });

      // Build active order parameters
      active_order_parameters = extract_active_order_parameter_ids(grains);
      unsigned int n_order_parameters = active_order_parameters.size();

      std::set<unsigned int> remap_candidates;

      // Base grain to compare with
      for (auto &[g_base_id, gr_base] : grains)
        {
          // Secondary grain
          for (auto &[g_other_id, gr_other] : grains)
            {
              if (g_other_id > g_base_id)
                {
                  if (gr_base.overlaps(gr_other,
                                       buffer_distance_ratio,
                                       buffer_distance_fixed))
                    {
                      dsp.add(grains_to_sparsity.at(g_base_id),
                              grains_to_sparsity.at(g_other_id));

                      // Exploit symmetry
                      dsp.add(grains_to_sparsity.at(g_other_id),
                              grains_to_sparsity.at(g_base_id));

                      if (gr_other.get_order_parameter_id() ==
                          gr_base.get_order_parameter_id())
                        {
                          remap_candidates.insert(
                            grains_to_sparsity.at(g_other_id));

                          if (do_logging)
                            {
                              std::ostringstream ss;
                              ss << "Found an overlap between grain "
                                 << gr_base.get_grain_id() << " and grain "
                                 << gr_other.get_grain_id()
                                 << " with order parameter "
                                 << gr_base.get_order_parameter_id()
                                 << std::endl;

                              log.emplace_back(ss.str());
                            }
                        }
                    }
                }
            }
        }

      const unsigned int n_collisions = remap_candidates.size();

      // We perform reassignment if collisions are detected or if we force it
      if (n_collisions > 0 || force_reassignment)
        {
          SparsityPattern sp;
          sp.copy_from(dsp);

          /* If we want to try at first a simplified remapping instead of
           * building the entire graph coloring. The disadvantage of the latter
           * approach is that it can lead to many subsequent costly remappings.
           * However, the simplified strategy may render a higher number of
           * order parameters in use. Anyway, if the simplified strategy renders
           * too large number of order parameters exceeding the maximum allowed
           * limit, the complete graph colorization is then attempted to be
           * performed. */
          bool perform_coloring = !try_fast_reassignment;

          if (try_fast_reassignment)
            {
              ScopedName sc("fast_reassignment");
              MyScope    scope(timer, sc, timer.is_enabled());

              /* Since this strategy may fail, the new order parameters are not
               * assigned directly to the grains but only gathered in a map. */
              std::map<unsigned int, unsigned int>
                new_order_parameters_for_grains;

              /* We also need to inverse the grains_to_sparsity map */
              std::map<unsigned int, unsigned int> sparsity_to_grains;

              std::transform(grains_to_sparsity.begin(),
                             grains_to_sparsity.end(),
                             std::inserter(sparsity_to_grains,
                                           sparsity_to_grains.end()),
                             [](const auto &a) {
                               return std::make_pair(a.second, a.first);
                             });

              for (const auto &row_id : remap_candidates)
                {
                  const auto grain_id = sparsity_to_grains.at(row_id);

                  auto available_colors = active_order_parameters;

                  for (auto it = sp.begin(row_id); it != sp.end(row_id); ++it)
                    {
                      const auto neighbor_id =
                        sparsity_to_grains.at(it->column());
                      const auto neighbor_order_parameter =
                        grains.at(neighbor_id).get_order_parameter_id();
                      available_colors.erase(neighbor_order_parameter);
                    }

                  unsigned int new_order_parameter;
                  if (!available_colors.empty())
                    {
                      new_order_parameter = *available_colors.begin();
                    }
                  else
                    {
                      new_order_parameter = n_order_parameters;
                      ++n_order_parameters;
                      active_order_parameters.insert(new_order_parameter);

                      if (n_order_parameters > max_order_parameters_num)
                        {
                          perform_coloring = true;

                          if (do_logging)
                            {
                              std::ostringstream ss;
                              ss
                                << "The simplified reassignment strategy exceeded "
                                << "the maximum number of order parameters allowed. "
                                << "Switching to the complete graph coloring."
                                << std::endl;
                              log.emplace_back(ss.str());
                            }

                          break;
                        }
                    }

                  if (grains.at(grain_id).get_order_parameter_id() !=
                      new_order_parameter)
                    new_order_parameters_for_grains[grain_id] =
                      new_order_parameter;
                }

              if (!perform_coloring && !new_order_parameters_for_grains.empty())
                {
                  for (const auto &[grain_id, new_order_parameter] :
                       new_order_parameters_for_grains)
                    grains.at(grain_id).set_order_parameter_id(
                      new_order_parameter);

                  grains_reassigned = true;
                }
            }

          if (perform_coloring)
            {
              ScopedName sc("full_reassignment");
              MyScope    scope(timer, sc, timer.is_enabled());

              std::vector<unsigned int> color_indices(n_grains);

              unsigned n_colors =
                SparsityTools::color_sparsity_pattern(sp, color_indices);
              AssertThrow(n_colors <= max_order_parameters_num,
                          ExcMessage(
                            "Maximum number of order parameters exceeded!"));

              for (auto &[gid, grain] : grains)
                {
                  const unsigned int new_order_parmeter =
                    color_indices[grains_to_sparsity.at(gid)] - 1;

                  if (grain.get_order_parameter_id() != new_order_parmeter)
                    {
                      grain.set_order_parameter_id(new_order_parmeter);
                      grains_reassigned = true;
                    }
                }

              // Rebuild completely active order parameters if remapping
              // has been performed via coloring
              active_order_parameters =
                extract_active_order_parameter_ids(grains);
              n_order_parameters = active_order_parameters.size();
            }
        }

      /* Build up neighbors connectivity. Depending on the regime chosen (prefer
       * closest or not) we use neighbors from different states for computing
       * the distance to the nearest one when determining the safe trasfer
       * buffer zone for remapping.
       */
      for (auto &[g_base_id, gr_base] : grains)
        {
          (void)g_base_id;
          for (const auto &[g_other_id, gr_other] : grains)
            {
              (void)g_other_id;
              if (gr_base.get_grain_id() != gr_other.get_grain_id() &&
                  (gr_base.get_order_parameter_id() ==
                     gr_other.get_order_parameter_id() ||
                   gr_base.get_old_order_parameter_id() ==
                     gr_other.get_old_order_parameter_id()))
                {
                  gr_base.add_neighbor(gr_other);
                }
            }
        }

      // Remove dangling order parameters if any
      const unsigned int max_order_parameter_id =
        *active_order_parameters.rbegin();
      const int mismatch = max_order_parameter_id - (n_order_parameters - 1);
      AssertThrow(mismatch >= 0,
                  ExcMessage("Error in active order parameters numbering!"));

      if (mismatch > 0)
        {
          std::map<unsigned int, int> ids_offsets;

          auto it = active_order_parameters.begin();

          for (unsigned int id = 0; id < n_order_parameters; ++id)
            {
              ids_offsets.emplace(*it, (*it) - id);
              it++;
            }

          for (auto &[gid, grain] : grains)
            {
              (void)gid;

              const unsigned int current_order_parameter_id =
                grain.get_order_parameter_id();

              if (ids_offsets.at(current_order_parameter_id) > 0)
                {
                  grain.set_order_parameter_id(
                    current_order_parameter_id -
                    ids_offsets.at(current_order_parameter_id));
                }
            }

          // If we are here, then for sure grains have been reassigned
          grains_reassigned = true;

          // Rebuild active order parameters
          active_order_parameters = extract_active_order_parameter_ids(grains);
        }

      if (do_logging)
        print_log(log);

      return std::make_tuple(n_collisions, grains_reassigned);
    }

    // Build inverse mapping for later use when forces and computed
    void
    build_inverse_mapping()
    {
      grain_segment_ids_numbering.clear();

      n_total_segments = 0;
      for (const auto &op_particle_ids : particle_ids_to_grain_ids)
        for (const auto &grain_and_segment : op_particle_ids)
          {
            grain_segment_ids_numbering[grain_and_segment.first]
                                       [grain_and_segment.second] =
                                         n_total_segments;

            ++n_total_segments;
          }
    }

    // Output particle ids
    void
    output_particle_ids(const LinearAlgebra::distributed::Vector<double>
                          &current_particle_ids) const
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = false;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      Vector<double> ranks(tria.n_active_cells());
      ranks = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      data_out.attach_triangulation(tria);

      data_out.add_data_vector(ranks,
                               "ranks",
                               DataOut<dim>::DataVectorType::type_cell_data);

      data_out.add_data_vector(current_particle_ids,
                               "particle_ids",
                               DataOut<dim>::DataVectorType::type_cell_data);

      data_out.build_patches();

      pcout << "Outputing particle_ids..." << std::endl;

      static unsigned int counter = 0;

      const std::string filename =
        "particle_ids." + std::to_string(counter) + ".vtu";
      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

      counter++;
    }

    // Output clouds as particles (fast)
    void
    output_grains(const std::map<unsigned int, Grain<dim>> &current_grains,
                  const std::string &                       prefix) const
    {
      /* The simplest mapping is provided since it is not employed by the
       * functionality used in this function. So we do not need here the
       * original mapping of the problem we are working with.
       */
      const MappingQ<dim> mapping(1);

      const unsigned int n_properties = 3;

      Particles::ParticleHandler particles_handler(
        dof_handler.get_triangulation(), mapping, n_properties);
      particles_handler.reserve(current_grains.size());

      const auto local_boxes = GridTools::compute_mesh_predicate_bounding_box(
        dof_handler.get_triangulation(), IteratorFilters::LocallyOwnedCell());
      const auto global_bounding_boxes =
        Utilities::MPI::all_gather(MPI_COMM_WORLD, local_boxes);

      std::vector<Point<dim>>          positions;
      std::vector<std::vector<double>> properties;

      // Append each cloud to the particle handler
      for (const auto &[gid, grain] : current_grains)
        {
          for (const auto &segment : grain.get_segments())
            {
              positions.push_back(segment.get_center());

              const unsigned int order_parameter_id =
                grain.get_order_parameter_id();
              properties.push_back(
                std::vector<double>({static_cast<double>(gid),
                                     segment.get_radius(),
                                     static_cast<double>(order_parameter_id)}));
            }
        }

      particles_handler.insert_global_particles(positions,
                                                global_bounding_boxes,
                                                properties);

      Particles::DataOut<dim>  particles_out;
      std::vector<std::string> data_component_names{"grain_id",
                                                    "radius",
                                                    "order_parameter"};
      particles_out.build_patches(particles_handler, data_component_names);

      pcout << "Outputing grains..." << std::endl;

      static unsigned int counter = 0;

      const std::string filename =
        prefix + "." + std::to_string(counter) + ".vtu";
      particles_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

      counter++;
    }

    // Print unique log events merged from multiple ranks
    void
    print_log(std::vector<std::string> &log) const
    {
      // Get all log entries
      auto all_logs = Utilities::MPI::gather(MPI_COMM_WORLD, log);

      // Identify unique remappings
      std::set<std::string> unique_events;
      for (auto &log_rank : all_logs)
        {
          std::copy(log_rank.begin(),
                    log_rank.end(),
                    std::inserter(unique_events, unique_events.end()));
        }

      // Print remapping events
      for (const auto &event : unique_events)
        {
          pcout << event;
        }
    }

    const DoFHandler<dim> &dof_handler;

    const parallel::TriangulationBase<dim> &tria;

    // Distributed vector of particle ids
    BlockVectorType op_particle_ids;

    // Mapping to find grain from particle id over the order paramter
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
      particle_ids_to_grain_ids;

    // The inverse mapping
    std::map<unsigned int, std::map<unsigned int, unsigned int>>
      grain_segment_ids_numbering;

    // Perform greedy initialization
    const bool greedy_init;

    // Are new grains allowed to emerge
    const bool allow_new_grains;

    // Use fast grains reassignment strategy
    const bool fast_reassignment;

    // Maximum number of order parameters available
    const unsigned int max_order_parameters_num;

    // Grain representation
    const GrainRepresentation grain_representation;

    // Minimum value of order parameter value
    const double threshold_lower;

    // Minimum threshold for the new grains
    const double threshold_new_grains;

    // Buffer zone around the grain - ratio value
    const double buffer_distance_ratio;

    // Buffer zone around the grain - fixed value
    const double buffer_distance_fixed;

    // Order parameters offset in FESystem
    const unsigned int order_parameters_offset;

    // Total number of segments
    unsigned int n_total_segments;

    // Use detailed logging
    const bool do_logging;

    // Use old remapping algo
    const bool use_old_remap;

    std::map<unsigned int, Grain<dim>> grains;
    std::map<unsigned int, Grain<dim>> old_grains;
    std::set<unsigned int>             active_order_parameters;

    ConditionalOStream pcout;

    const double invalid_particle_id = -1.0;

    mutable MyTimerOutput timer;
  };
} // namespace GrainTracker
