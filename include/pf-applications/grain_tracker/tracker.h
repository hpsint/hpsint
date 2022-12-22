#pragma once

#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <functional>

#include "distributed_stitching.h"
#include "grain.h"
#include "periodicity_graph.h"
#include "remap_graph.h"
#include "remapping.h"
#include "segment.h"

#define AssertThrowDistributedDimension(size)                        \
  {                                                                  \
    const auto min_size = Utilities::MPI::min(size, MPI_COMM_WORLD); \
    const auto max_size = Utilities::MPI::max(size, MPI_COMM_WORLD); \
    AssertThrow(min_size == max_size,                                \
                ExcDimensionMismatch(min_size, max_size));           \
  }

namespace GrainTracker
{
  using namespace dealii;

  DeclExceptionMsg(ExcGrainsInconsistency, "Grains inconsistency detected!");

  /* The grain tracker algo itself. */
  template <int dim, typename Number>
  class Tracker
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    Tracker(const DoFHandler<dim> &                 dof_handler,
            const parallel::TriangulationBase<dim> &tria,
            const bool                              greedy_init,
            const bool                              allow_new_grains,
            const unsigned int                      max_order_parameters_num,
            const double                            threshold_lower = 0.01,
            const double                            threshold_upper = 1.01,
            const double       buffer_distance_ratio                = 0.05,
            const unsigned int op_offset                            = 2)
      : dof_handler(dof_handler)
      , tria(tria)
      , greedy_init(greedy_init)
      , allow_new_grains(allow_new_grains)
      , max_order_parameters_num(max_order_parameters_num)
      , threshold_lower(threshold_lower)
      , threshold_upper(threshold_upper)
      , buffer_distance_ratio(buffer_distance_ratio)
      , order_parameters_offset(op_offset)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    /* Track grains over timesteps. The function returns a tuple of bool
     * variables which signify if any grains have been reassigned and if the
     * number of active order parameters has been changed.
     */
    std::tuple<bool, bool>
    track(const BlockVectorType &solution, const unsigned int n_order_params)
    {
      // Copy old grains
      old_grains = grains;
      grains.clear();

      // Now we do not assign grain indices when searching for grains
      const bool assign_indices = false;

      const auto new_grains =
        detect_grains(solution, n_order_params, assign_indices);

      // Numberer for new grains
      unsigned int grain_numberer = old_grains.rbegin()->first + 1;

      // Create a list of grain candidates
      std::set<unsigned int> grains_candidates;
      for (const auto &[gid, gr] : old_grains)
        {
          grains_candidates.insert(gid);
        }

      // Create map with the remapped particles
      std::vector<std::vector<bool>> grains_remapped(
        particle_ids_to_grain_ids.size());

      for (unsigned int iop = 0; iop < particle_ids_to_grain_ids.size(); ++iop)
        grains_remapped[iop].assign(particle_ids_to_grain_ids[iop].size(),
                                    false);

      // Create segments and transfer grain_id's for them
      for (const auto &[current_grain_id, new_grain] : new_grains)
        {
          (void)current_grain_id;

          /* Search for an old segment closest to the new one and get its grain
           * id, this will be assigned the new segment.
           */
          double       min_distance = std::numeric_limits<double>::max();
          unsigned int new_grain_id = std::numeric_limits<unsigned int>::max();

          for (const auto &new_segment : new_grain.get_segments())
            {
              for (const auto &old_grain_id : grains_candidates)
                {
                  const auto &old_grain = old_grains.at(old_grain_id);

                  for (const auto &old_segment : old_grain.get_segments())
                    {
                      const double distance = new_segment.get_center().distance(
                        old_segment.get_center());

                      if (distance < new_segment.get_radius() &&
                          distance < min_distance)
                        {
                          min_distance = distance;
                          new_grain_id = old_grain.get_grain_id();
                        }
                    }
                }
            }

          // Set up the grain number
          if (new_grain_id == std::numeric_limits<unsigned int>::max())
            {
              if (allow_new_grains)
                {
                  new_grain_id = grain_numberer++;
                }
              else
                {
                  // Check if we have found anything
                  AssertThrow(
                    new_grain_id != std::numeric_limits<unsigned int>::max(),
                    ExcGrainsInconsistency(
                      "Unable to match a new grain with an old one from the previous configuration!"));
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
                  std::string("\r\n    min_distance                = ") +
                  std::to_string(min_distance)
              ));
              // clang-format on

              grains_candidates.erase(new_grain_id);
            }

          // Insert new grain
          grains.emplace(std::make_pair(new_grain_id, new_grain));
          grains.at(new_grain_id).set_grain_id(new_grain_id);

          // Update mapping since we changed the grain id
          if (new_grain_id != current_grain_id)
            {
              auto &particle_to_grain =
                particle_ids_to_grain_ids[new_grain.get_order_parameter_id()];

              for (unsigned int ip = 0; ip < particle_to_grain.size(); ip++)
                {
                  auto &pmap = particle_to_grain[ip];

                  if (grains_remapped[new_grain.get_order_parameter_id()][ip] ==
                        false &&
                      pmap.first == current_grain_id)
                    {
                      pmap.first = new_grain_id;
                      grains_remapped[new_grain.get_order_parameter_id()][ip] =
                        true;
                    }
                }
            }
        }

      /* For tracking we want the grains assigned to the same order parameter to
       * be as far from each other as possible to reduce the number of costly
       * grains reassignment.
       */
      const bool force_reassignment = false;

      // Reassign grains
      const bool grains_reassigned = reassign_grains(force_reassignment);

      // Check if number of order parameters has changed
      bool op_number_changed =
        (active_order_parameters.size() !=
           build_old_order_parameter_ids(grains).size() ||
         active_order_parameters.size() !=
           build_active_order_parameter_ids(old_grains).size());

      return std::make_tuple(grains_reassigned, op_number_changed);
    }

    /* Initialization of grains at the very first step. The function returns a
     * tuple of bool variables which signify if any grains have been reassigned
     * and if the number of active order parameters has been changed.
     */
    std::tuple<bool, bool>
    initial_setup(const BlockVectorType &solution,
                  const unsigned int     n_order_params)
    {
      const bool assign_indices = true;

      grains = detect_grains(solution, n_order_params, assign_indices);

      // The rest is the same as was before

      /* Initial grains reassignment, the closest neighbors are allowed as we
       * want to minimize the number of order parameters in use.
       */
      const bool force_reassignment = greedy_init;

      // Reassign grains
      const bool grains_reassigned = reassign_grains(force_reassignment);

      // Check if number of order parameters has changed
      const bool op_number_changed =
        (active_order_parameters.size() !=
         build_old_order_parameter_ids(grains).size());

      return std::make_tuple(grains_reassigned, op_number_changed);
    }

    // Remap a single state vector
    void
    remap(BlockVectorType &solution) const
    {
      remap({&solution});
    }

    // Remap state vectors
    void
    remap(std::vector<BlockVectorType *> solutions) const
    {
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

      for (const auto &[gid, gr] : disappered_grains)
        {
          const unsigned int op_id =
            gr.get_order_parameter_id() + order_parameters_offset;

          std::ostringstream ss;
          ss << "Grain " << gr.get_grain_id() << " having order parameter "
             << op_id << " has disappered" << std::endl;
          log.emplace_back(ss.str());

          alter_dof_values_for_grain(
            gr,
            [this,
             &values,
             op_id](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                    BlockVectorType *                               solution) {
              cell.get_dof_values(solution->block(op_id), values);
              values = 0;
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
          /* Check if the graph has cycles - these are unlikely situations and
           * at the moment we do not handle them due to complexity.
           */

          std::ostringstream ss;
          ss << "Remapping dependencies have been detected and resolved."
             << std::endl;
          graph.print(ss);
          log.emplace_back(ss.str());

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

          std::ostringstream ss;
          ss << "Remap order parameter for grain id = " << re.grain_id
             << ": from " << re.from << " to temp" << std::endl;
          log.emplace_back(ss.str());

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
          alter_dof_values_for_grain(
            grain,
            [this,
             &values,
             op_id_src](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                        BlockVectorType *solution) {
              cell.get_dof_values(solution->block(op_id_src), values);
              values = 0;
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

          std::ostringstream ss;
          ss << "Remap order parameter for grain id = " << re.grain_id
             << ": from " << re.from << " to " << re.to << std::endl;
          log.emplace_back(ss.str());

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
          alter_dof_values_for_grain(
            grain,
            [this,
             &values,
             op_id_src](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                        BlockVectorType *solution) {
              cell.get_dof_values(solution->block(op_id_src), values);
              values = 0;
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

          std::ostringstream ss;
          ss << "Remap order parameter for grain id = " << re.grain_id
             << ": from temp to " << re.to << std::endl;
          log.emplace_back(ss.str());

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

      print_log(log);
    }

    const std::map<unsigned int, Grain<dim>> &
    get_grains() const
    {
      return grains;
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

    unsigned int
    get_particle_index(const unsigned int order_parameter,
                       const unsigned int cell_index) const
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

    std::pair<unsigned int, unsigned int>
    get_grain_and_segment(const unsigned int order_parameter,
                          const unsigned int particle_id) const
    {
      AssertThrow(particle_id != invalid_particle_id,
                  ExcMessage("Invalid particle_id provided"));

      return particle_ids_to_grain_ids[order_parameter][particle_id];
    }

    const Point<dim> &
    get_segment_center(const unsigned int grain_id,
                       const unsigned int segment_id) const
    {
      return grains.at(grain_id).get_segments()[segment_id].get_center();
    }

    unsigned int
    get_grain_segment_index(const unsigned int grain_id,
                            const unsigned int segment_id) const
    {
      return grain_segment_ids_numbering.at(grain_id).at(segment_id);
    }

    unsigned int
    n_segments() const
    {
      return n_total_segments;
    }

  private:
    unsigned int
    run_flooding(const typename DoFHandler<dim>::cell_iterator &cell,
                 const BlockVectorType &                        solution,
                 LinearAlgebra::distributed::Vector<Number> &   particle_ids,
                 const unsigned int order_parameter_id,
                 const unsigned int id)
    {
      if (cell->has_children())
        {
          unsigned int counter = 0;

          for (const auto &child : cell->child_iterators())
            counter += run_flooding(
              child, solution, particle_ids, order_parameter_id, id);

          return counter;
        }

      if (cell->is_locally_owned() == false)
        return 0;

      const auto particle_id = particle_ids[cell->global_active_cell_index()];

      if (particle_id != invalid_particle_id)
        return 0; // cell has been visited

      Vector<double> values(cell->get_fe().n_dofs_per_cell());

      cell->get_dof_values(solution.block(order_parameter_id +
                                          order_parameters_offset),
                           values);

      if (values.linfty_norm() < threshold_lower)
        return 0; // cell has no particle

      particle_ids[cell->global_active_cell_index()] = id;

      unsigned int counter = 1;

      for (const auto face : cell->face_indices())
        if (cell->at_boundary(face) == false)
          counter += run_flooding(cell->neighbor(face),
                                  solution,
                                  particle_ids,
                                  order_parameter_id,
                                  id);

      return counter;
    }

    std::map<unsigned int, Grain<dim>>
    detect_grains(const BlockVectorType &solution,
                  const unsigned int     n_order_params,
                  const bool             assign_indices)
    {
      std::map<unsigned int, Grain<dim>> new_grains;

      const MPI_Comm comm = MPI_COMM_WORLD;

      // Numerator
      unsigned int grains_numerator = 0;

      // Order parameter indices stored per cell
      op_particle_ids.reinit(n_order_params);
      for (unsigned int b = 0; b < op_particle_ids.n_blocks(); ++b)
        op_particle_ids.block(b).reinit(
          tria.global_active_cell_index_partitioner().lock());

      particle_ids_to_grain_ids.clear();
      particle_ids_to_grain_ids.resize(n_order_params);

      for (unsigned int current_order_parameter_id = 0;
           current_order_parameter_id < n_order_params;
           ++current_order_parameter_id)
        {
          auto &particle_ids =
            op_particle_ids.block(current_order_parameter_id);

          // step 1) run flooding and determine local particles and give them
          // local ids
          particle_ids = invalid_particle_id;

          unsigned int counter = 0;
          unsigned int offset  = 0;

          for (const auto &cell : dof_handler.active_cell_iterators())
            if (run_flooding(cell,
                             solution,
                             particle_ids,
                             current_order_parameter_id,
                             counter) > 0)
              counter++;

          // step 2) determine the global number of locally determined particles
          // and give each one an unique id by shifting the ids
          MPI_Exscan(&counter, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

          for (auto &particle_id : particle_ids)
            if (particle_id != invalid_particle_id)
              particle_id += offset;

          // step 3) get particle ids on ghost cells and figure out if local
          // particles and ghost particles might be one particle
          particle_ids.update_ghost_values();

          std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
            local_connectiviy(counter);

          for (const auto &ghost_cell :
               dof_handler.get_triangulation().active_cell_iterators())
            if (ghost_cell->is_ghost())
              {
                const auto particle_id =
                  particle_ids[ghost_cell->global_active_cell_index()];

                if (particle_id == invalid_particle_id)
                  continue;

                for (const auto face : ghost_cell->face_indices())
                  {
                    if (ghost_cell->at_boundary(face))
                      continue;

                    const auto add = [&](const auto &ghost_cell,
                                         const auto &local_cell) {
                      if (local_cell->is_locally_owned() == false)
                        return;

                      const auto neighbor_particle_id =
                        particle_ids[local_cell->global_active_cell_index()];

                      if (neighbor_particle_id == invalid_particle_id)
                        return;

                      auto &temp =
                        local_connectiviy[neighbor_particle_id - offset];
                      temp.emplace_back(ghost_cell->subdomain_id(),
                                        particle_id);
                      std::sort(temp.begin(), temp.end());
                      temp.erase(std::unique(temp.begin(), temp.end()),
                                 temp.end());
                    };

                    if (ghost_cell->neighbor(face)->has_children())
                      {
                        for (unsigned int subface = 0;
                             subface <
                             GeometryInfo<dim>::n_subfaces(
                               internal::SubfaceCase<dim>::case_isotropic);
                             ++subface)
                          add(ghost_cell,
                              ghost_cell->neighbor_child_on_subface(face,
                                                                    subface));
                      }
                    else
                      add(ghost_cell, ghost_cell->neighbor(face));
                  }
              }

          // step 4) based on the local-ghost information, figure out all
          // particles on all processes that belong togher (unification ->
          // clique), give each clique an unique id, and return mapping from the
          // global non-unique ids to the global ids
          const auto local_to_global_particle_ids =
            perform_distributed_stitching(comm, local_connectiviy);

          // step 5) determine properties of particles (volume, radius, center)
          unsigned int n_particles = 0;

          // ... determine the number of particles
          if (Utilities::MPI::sum(local_to_global_particle_ids.size(), comm) ==
              0)
            n_particles = 0;
          else
            {
              n_particles =
                (local_to_global_particle_ids.size() == 0) ?
                  0 :
                  *std::max_element(local_to_global_particle_ids.begin(),
                                    local_to_global_particle_ids.end());
              n_particles = Utilities::MPI::max(n_particles, comm) + 1;
            }

          std::vector<double> particle_info(n_particles * (1 + dim));

          // ... compute local information
          for (const auto &cell :
               dof_handler.get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto particle_id =
                  particle_ids[cell->global_active_cell_index()];

                if (particle_id == invalid_particle_id)
                  continue;

                const unsigned int unique_id = local_to_global_particle_ids
                  [static_cast<unsigned int>(particle_id) - offset];

                AssertIndexRange(unique_id, n_particles);

                particle_info[(dim + 1) * unique_id + 0] += cell->measure();

                for (unsigned int d = 0; d < dim; ++d)
                  particle_info[(dim + 1) * unique_id + 1 + d] +=
                    cell->center()[d] * cell->measure();
              }

          // ... reduce information
          MPI_Allreduce(MPI_IN_PLACE,
                        particle_info.data(),
                        particle_info.size(),
                        MPI_DOUBLE,
                        MPI_SUM,
                        comm);

          // ... compute particles centers
          std::vector<Point<dim>> particle_centers(n_particles);
          for (unsigned int i = 0; i < n_particles; i++)
            {
              for (unsigned int d = 0; d < dim; ++d)
                {
                  particle_centers[i][d] =
                    particle_info[i * (1 + dim) + 1 + d] /
                    particle_info[i * (1 + dim)];
                }
            }

          // ... compute particles radii
          std::vector<double> particle_radii(n_particles, 0.);
          for (const auto &cell :
               dof_handler.get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto particle_id =
                  particle_ids[cell->global_active_cell_index()];

                if (particle_id == invalid_particle_id)
                  continue;

                const unsigned int unique_id = local_to_global_particle_ids
                  [static_cast<unsigned int>(particle_id) - offset];

                AssertIndexRange(unique_id, n_particles);

                const auto &center = particle_centers[unique_id];

                const double dist =
                  center.distance(cell->barycenter()) + cell->diameter() / 2.;
                particle_radii[unique_id] =
                  std::max(particle_radii[unique_id], dist);
              }

          // ... reduce information
          MPI_Allreduce(MPI_IN_PLACE,
                        particle_radii.data(),
                        particle_radii.size(),
                        MPI_DOUBLE,
                        MPI_MAX,
                        comm);

          // Set global ids to the particles
          for (auto &particle_id : particle_ids)
            if (particle_id != invalid_particle_id)
              particle_id = local_to_global_particle_ids
                [static_cast<unsigned int>(particle_id) - offset];
          particle_ids.update_ghost_values();

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
                             subface <
                             GeometryInfo<dim>::n_subfaces(
                               internal::SubfaceCase<dim>::case_isotropic);
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
            Utilities::MPI::all_gather(MPI_COMM_WORLD, periodicity_flatten);

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

          // Parse groups at first to create grains
          for (unsigned int i = 0; i < n_particles; ++i)
            {
              if (particle_groups[i] != numbers::invalid_unsigned_int)
                {
                  unsigned int grain_id = particle_groups[i] + grains_numerator;

                  new_grains.try_emplace(grain_id,
                                         assign_indices ?
                                           grain_id :
                                           numbers::invalid_unsigned_int,
                                         current_order_parameter_id);

                  new_grains.at(grain_id).add_segment(particle_centers[i],
                                                      particle_radii[i]);

                  free_particles.erase(i);

                  const unsigned int last_segment_id =
                    new_grains.at(grain_id).n_segments() - 1;

                  particle_ids_to_grain_ids[current_order_parameter_id][i] =
                    std::make_pair(grain_id, last_segment_id);
                }
            }

          grains_numerator += n_groups_found;

          // Then handle the remaining non-periodic particles
          for (const unsigned int i : free_particles)
            {
              unsigned int grain_id = grains_numerator;

              new_grains.try_emplace(grain_id,
                                     assign_indices ?
                                       grain_id :
                                       numbers::invalid_unsigned_int,
                                     current_order_parameter_id);

              new_grains.at(grain_id).add_segment(particle_centers[i],
                                                  particle_radii[i]);

              const unsigned int last_segment_id =
                new_grains.at(grain_id).n_segments() - 1;

              particle_ids_to_grain_ids[current_order_parameter_id][i] =
                std::make_pair(grain_id, last_segment_id);

              ++grains_numerator;
            }
        }

      // Build inverse mapping for later use when forces and computed
      n_total_segments = 0;
      for (const auto &op_particle_ids : particle_ids_to_grain_ids)
        for (const auto &grain_and_segment : op_particle_ids)
          {
            grain_segment_ids_numbering[grain_and_segment.first]
                                       [grain_and_segment.second] =
                                         n_total_segments;

            ++n_total_segments;
          }

      return new_grains;
    }

    // Build a set of active order parameters
    std::set<unsigned int>
    build_active_order_parameter_ids(
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

    // Build a set of old order parameters
    std::set<unsigned int>
    build_old_order_parameter_ids(
      const std::map<unsigned int, Grain<dim>> &all_grains) const
    {
      std::set<unsigned int> old_op_ids;

      for (const auto &[gid, gr] : all_grains)
        {
          (void)gid;
          old_op_ids.insert(gr.get_old_order_parameter_id());
        }

      return old_op_ids;
    }

    // Reassign grains order parameters to prevent collision
    bool
    reassign_grains(const bool force_reassignment)
    {
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

      /* If we force grains reassignment, then we set up this flag so the
       * colorization algorithm is forced to be executed
       */
      bool overlap_detected = force_reassignment;

      // Base grain to compare with
      for (auto &[g_base_id, gr_base] : grains)
        {
          // Secondary grain
          for (auto &[g_other_id, gr_other] : grains)
            {
              if (g_other_id != g_base_id)
                {
                  // Minimum distance between the two grains
                  double min_distance = gr_base.distance(gr_other);

                  /* Buffer safety zone around the two grains. If an overlap
                   * is detected, then the old order parameter values of all
                   * the cells inside the buffer zone are transfered to a
                   * new one.
                   */
                  double buffer_distance_base =
                    buffer_distance_ratio * gr_base.get_max_radius();
                  double buffer_distance_other =
                    buffer_distance_ratio * gr_other.get_max_radius();

                  /* If two grains sharing the same order parameter are
                   * too close to each other, then try to change the
                   * order parameter of the secondary grain
                   */
                  if (min_distance <
                      buffer_distance_base + buffer_distance_other)
                    {
                      dsp.add(grains_to_sparsity.at(g_base_id),
                              grains_to_sparsity.at(g_other_id));

                      if (gr_other.get_order_parameter_id() ==
                          gr_base.get_order_parameter_id())
                        {
                          std::ostringstream ss;
                          ss << "Found an overlap between grain "
                             << gr_base.get_grain_id() << " and grain "
                             << gr_other.get_grain_id()
                             << " with order parameter "
                             << gr_base.get_order_parameter_id() << std::endl;

                          log.emplace_back(ss.str());

                          overlap_detected = true;
                        }
                    }
                }
            }
        }

      if (overlap_detected)
        {
          SparsityPattern sp;
          sp.copy_from(dsp);

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
                  gr_base.add_neighbor(&gr_other);
                }
            }
        }

      // Build active order parameters
      active_order_parameters = build_active_order_parameter_ids(grains);

      // Remove dangling order parameters if any
      const unsigned int n_order_parameters = active_order_parameters.size();
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

          // If we are here, then for sure grains has been reassigned
          grains_reassigned = true;

          // Rebuild active order parameters
          active_order_parameters = build_active_order_parameter_ids(grains);
        }

      print_log(log);

      return grains_reassigned;
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

    // Print current grains
    template <typename Stream>
    void
    print_grains(const std::map<unsigned int, Grain<dim>> &current_grains,
                 Stream &                                  out) const
    {
      out << "Number of order parameters: "
          << build_active_order_parameter_ids(current_grains).size()
          << std::endl;
      out << "Number of grains: " << current_grains.size() << std::endl;
      for (const auto &[gid, gr] : current_grains)
        {
          (void)gid;
          out << "op_index_current = " << gr.get_order_parameter_id()
              << " | op_index_old = " << gr.get_old_order_parameter_id()
              << " | segments = " << gr.get_segments().size()
              << " | grain_index = " << gr.get_grain_id() << std::endl;
          for (const auto &segment : gr.get_segments())
            {
              out << "    segment: center = " << segment.get_center()
                  << " | radius = " << segment.get_radius() << std::endl;
            }
        }
    }

    // Print current grains ordered according to segments location
    template <typename Stream>
    void
    print_grains_invariant(
      const std::map<unsigned int, Grain<dim>> &current_grains,
      Stream &                                  out) const
    {
      std::vector<unsigned int>                         ordered_grains;
      std::map<unsigned int, std::vector<unsigned int>> ordered_segments;

      for (const auto &pair_gid_grain : current_grains)
        {
          const auto &grain_id = pair_gid_grain.first;
          const auto &grain    = pair_gid_grain.second;

          ordered_grains.push_back(grain_id);

          ordered_segments.emplace(grain_id, std::vector<unsigned int>());
          for (unsigned int i = 0; i < grain.get_segments().size(); i++)
            {
              ordered_segments.at(grain_id).push_back(i);
            }

          std::sort(
            ordered_segments.at(grain_id).begin(),
            ordered_segments.at(grain_id).end(),
            [&grain](const auto &segment_a_id, const auto &segment_b_id) {
              const auto &segment_a = grain.get_segments()[segment_a_id];
              const auto &segment_b = grain.get_segments()[segment_b_id];

              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (segment_a.get_center()[d] != segment_b.get_center()[d])
                    {
                      return segment_a.get_center()[d] <
                             segment_b.get_center()[d];
                    }
                }
              return false;
            });
        }

      std::sort(
        ordered_grains.begin(),
        ordered_grains.end(),
        [&current_grains, &ordered_segments](const auto &grain_a_id,
                                             const auto &grain_b_id) {
          const auto &grain_a = current_grains.at(grain_a_id);
          const auto &grain_b = current_grains.at(grain_b_id);

          const auto &min_segment_a =
            grain_a
              .get_segments()[ordered_segments.at(grain_a.get_grain_id())[0]];
          const auto &min_segment_b =
            grain_b
              .get_segments()[ordered_segments.at(grain_b.get_grain_id())[0]];

          for (unsigned int d = 0; d < dim; ++d)
            {
              if (min_segment_a.get_center()[d] !=
                  min_segment_b.get_center()[d])
                {
                  return min_segment_a.get_center()[d] <
                         min_segment_b.get_center()[d];
                }
            }
          return false;
        });

      // Printing itself
      out << "Number of order parameters: "
          << build_active_order_parameter_ids(current_grains).size()
          << std::endl;
      out << "Number of grains: " << current_grains.size() << std::endl;

      for (const auto &grain_id : ordered_grains)
        {
          const auto &grain = current_grains.at(grain_id);

          out << "op_index_current = " << grain.get_order_parameter_id()
              << " | op_index_old = " << grain.get_old_order_parameter_id()
              << " | segments = " << grain.get_segments().size() << std::endl;

          for (const auto &segment_id : ordered_segments.at(grain_id))
            {
              const auto &segment =
                current_grains.at(grain_id).get_segments()[segment_id];

              out << "    segment: center = " << segment.get_center()
                  << " | radius = " << segment.get_radius() << std::endl;
            }
        }
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

    // Maximum number of order parameters available
    const unsigned int max_order_parameters_num;

    // Minimum value of order parameter value
    const double threshold_lower;

    // Maximum value of order parameter value
    const double threshold_upper;

    // Buffer zone around the grain
    const double buffer_distance_ratio;

    // Order parameters offset in FESystem
    const unsigned int order_parameters_offset;

    // Total number of segments
    unsigned int n_total_segments;

    std::map<unsigned int, Grain<dim>> grains;
    std::map<unsigned int, Grain<dim>> old_grains;
    std::set<unsigned int>             active_order_parameters;

    ConditionalOStream pcout;

    const double invalid_particle_id = -1.0;
  };
} // namespace GrainTracker
