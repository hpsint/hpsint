#pragma once

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <functional>

#include "cloud.h"
#include "grain.h"
#include "remap_graph.h"
#include "remapping.h"
#include "segment.h"

namespace GrainTracker
{
  using namespace dealii;

  DeclExceptionMsg(ExcCloudsInconsistency, "Clouds inconsistency detected!");

  /* The grain tracker algo itself. */
  template <int dim, typename Number>
  class Tracker
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    Tracker(const DoFHandler<dim> &dof_handler,
            const bool             greedy_init,
            const double           threshold_lower       = 0.01,
            const double           threshold_upper       = 1.01,
            const double           buffer_distance_ratio = 0.05,
            const unsigned int     default_op_id         = 0,
            const unsigned int     op_offset             = 2)
      : dof_handler(dof_handler)
      , greedy_init(greedy_init)
      , threshold_lower(threshold_lower)
      , threshold_upper(threshold_upper)
      , buffer_distance_ratio(buffer_distance_ratio)
      , default_order_parameter_id(default_op_id)
      , order_parameters_offset(op_offset)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {}

    /* Track grains over timesteps. The function returns a tuple of bool
     * variables which signify if any grains have been reassigned and if the
     * number of active order parameters has been changed.
     */
    std::tuple<bool, bool>
    track(const BlockVectorType &solution)
    {
      // Find cells clouds
      last_clouds = std::move(find_clouds(solution));

      // Copy old grains
      old_grains = grains;

      // Clear current grains
      grains.clear();

      // Create segments and transfer grain_id's for them
      for (auto &cloud : last_clouds)
        {
          // New segment
          Segment<dim> current_segment(cloud);

          /* Search for an old segment closest to the new one and get its grain
           * id, this will be assigned the new segment.
           */
          double       min_distance = std::numeric_limits<double>::max();
          unsigned int grain_index_at_min_distance =
            std::numeric_limits<unsigned int>::max();

          for (const auto &[gid, gr] : old_grains)
            {
              (void)gid;

              for (const auto &segment : gr.get_segments())
                {
                  const double distance =
                    current_segment.get_center().distance(segment.get_center());

                  if (distance < min_distance)
                    {
                      min_distance                = distance;
                      grain_index_at_min_distance = gr.get_grain_id();
                    }
                }
            }

          // TODO: limit the maximum value of min_distance to prevent from
          // assigning the grain to some very distant one

          AssertThrow(
            grain_index_at_min_distance !=
              std::numeric_limits<unsigned int>::max(),
            ExcMessage(
              "Unable to detect a segment from the previous configuration for the cloud!"));

          // clang-format off
          AssertThrow(old_grains.at(grain_index_at_min_distance).get_order_parameter_id() 
            == cloud.get_order_parameter_id(),
            ExcCloudsInconsistency(
              std::string("Something got wrong with the order parameters numbering:\r\n") +
              std::string("\r\n    grain_index_at_min_distance = ") +
              std::to_string(grain_index_at_min_distance) + 
              std::string("\r\n    old grain order parameter   = ") +
              std::to_string(old_grains.at(grain_index_at_min_distance).get_order_parameter_id()) + 
              std::string("\r\n    cloud order parameter       = ") +
              std::to_string(cloud.get_order_parameter_id()) + 
              std::string("\r\n    min_distance                = ") +
              std::to_string(min_distance)
          ));
          // clang-format on

          grains.try_emplace(grain_index_at_min_distance,
                             grain_index_at_min_distance,
                             old_grains.at(grain_index_at_min_distance)
                               .get_order_parameter_id());

          grains.at(grain_index_at_min_distance).add_segment(current_segment);
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
    initial_setup(const BlockVectorType &solution)
    {
      /* We can perform initialization in two ways: greedy or not. Greedy
       * initialization signals that we have to reassign grains in optimal way
       * regardless on how they were defined by the initial conditions. PBC
       * information is used to identify grains consisting of multiple segments.
       */

      // Store last clouds state while iterating over groups
      last_clouds.clear();

      const auto grouped_clouds = find_grouped_clouds(solution);

      unsigned int grain_numberer = 0;
      for (auto &group : grouped_clouds)
        {
          const unsigned int grain_id = grain_numberer;

          for (auto &cloud : group)
            {
              const unsigned int order_parameter_id =
                cloud.get_order_parameter_id();
              grains.try_emplace(grain_id, grain_id, order_parameter_id);

              Segment<dim> segment(cloud);
              grains.at(grain_id).add_segment(segment);

              last_clouds.emplace_back(std::move(cloud));
            }

          grain_numberer++;
        }

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

          double transfer_buffer = grain.transfer_buffer();

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
          const unsigned int op_id = gr.get_order_parameter_id();

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
              cell.get_dof_values(solution->block(op_id +
                                                  order_parameters_offset),
                                  values);
              values = 0;
              cell.set_dof_values(values,
                                  solution->block(op_id +
                                                  order_parameters_offset));
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

      /* If graph is not empty, then have some dependencies in remapping and
       * need to perform at first those at the end of the graph in order not to
       * break the configuration of the domain.
       */
      if (!graph.empty())
        {
          /* Check if the graph has cycles - these are unlikely situations and
           * at the moment we do not handle them due to complexity.
           *
           * TODO: Resolve cycles in remappings
           */
          AssertThrow(graph.has_cycles() == false,
                      ExcMessage("Cycles detected in remappings!"));

          std::ostringstream ss;
          ss << "Remapping dependencies have been detected and resolved."
             << std::endl;
          graph.print(ss);
          log.emplace_back(ss.str());

          graph.rearrange(remappings);
        }

      // Now transfer values for the remaining grains
      for (const auto &re : remappings)
        {
          const auto &grain = grains.at(re.grain_id);

          /* Transfer buffer is the extra zone around the grain within which
           * the order parameters are swapped. Its maximum size is the half
           * of the distance to the nearest neighbor.
           */
          const unsigned int op_id_src = re.from;
          const unsigned int op_id_dst = re.to;

          std::ostringstream ss;
          ss << "Remap order parameter for grain id = " << re.grain_id
             << ": from " << op_id_src << " to " << op_id_dst << std::endl;
          log.emplace_back(ss.str());

          /* At first we transfer the values from the dofs related to the
           * old order parameters to the dofs of the new order parameter.
           */
          alter_dof_values_for_grain(
            grain,
            [this, &values, op_id_src, op_id_dst](
              const dealii::DoFCellAccessor<dim, dim, false> &cell,
              BlockVectorType *                               solution) {
              cell.get_dof_values(solution->block(op_id_src +
                                                  order_parameters_offset),
                                  values);
              cell.set_dof_values(values,
                                  solution->block(op_id_dst +
                                                  order_parameters_offset));
            });

          // Then we iterate again to nullify the old dofs
          alter_dof_values_for_grain(
            grain,
            [this,
             &values,
             op_id_src](const dealii::DoFCellAccessor<dim, dim, false> &cell,
                        BlockVectorType *solution) {
              cell.get_dof_values(solution->block(op_id_src +
                                                  order_parameters_offset),
                                  values);
              values = 0;
              cell.set_dof_values(values,
                                  solution->block(op_id_src +
                                                  order_parameters_offset));
            });
        }

      print_log(log);
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
    print_current_grains(Stream &out) const
    {
      print_grains(grains, out);
    }

    // Print last grains
    template <typename Stream>
    void
    print_old_grains(Stream &out) const
    {
      print_grains(old_grains, out);
    }

    // Output last set of clouds
    void
    dump_last_clouds() const
    {
      print_old_grains(pcout);
      print_clouds(last_clouds, pcout);
      output_clouds(last_clouds, /*is_merged = */ true);
    }

  private:
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

    // Find cells clouds
    std::vector<Cloud<dim>>
    find_clouds(const BlockVectorType &solution)
    {
      std::vector<Cloud<dim>> clouds;

      const unsigned int n_order_params =
        solution.n_blocks() - order_parameters_offset;

      for (unsigned int current_order_parameter_id = 0;
           current_order_parameter_id < n_order_params;
           ++current_order_parameter_id)
        {
          auto op_clouds =
            find_clouds_for_order_parameter(solution,
                                            current_order_parameter_id);
          clouds.insert(clouds.end(), op_clouds.begin(), op_clouds.end());
        }

      return clouds;
    }

    // Find cells clouds
    std::vector<std::vector<Cloud<dim>>>
    find_grouped_clouds(const BlockVectorType &solution)
    {
      std::vector<std::vector<Cloud<dim>>> clouds_groups;

      const unsigned int n_order_params =
        solution.n_blocks() - order_parameters_offset;

      for (unsigned int current_order_parameter_id = 0;
           current_order_parameter_id < n_order_params;
           ++current_order_parameter_id)
        {
          auto op_clouds =
            find_clouds_for_order_parameter(solution,
                                            current_order_parameter_id);

          /* Split into grains and also detect periodicity */
          std::vector<std::vector<Cloud<dim>>> op_clouds_groups;
          for (; !op_clouds.empty(); op_clouds.pop_back())
            {
              auto &cloud_current = op_clouds.back();

              if (cloud_current.has_periodic_boundary())
                {
                  bool periodic_found = false;

                  for (auto &group : op_clouds_groups)
                    {
                      for (const auto &cloud_secondary : group)
                        {
                          if (cloud_secondary.has_periodic_boundary() &&
                              cloud_secondary.is_periodic_with(cloud_current))
                            {
                              group.emplace_back(std::move(cloud_current));
                              periodic_found = true;
                              break;
                            }
                        }

                      if (periodic_found)
                        {
                          break;
                        }
                    }

                  if (!periodic_found)
                    {
                      op_clouds_groups.emplace_back(std::vector<Cloud<dim>>());
                      op_clouds_groups.back().emplace_back(
                        std::move(cloud_current));
                    }
                }
              else
                {
                  op_clouds_groups.emplace_back(std::vector<Cloud<dim>>());
                  op_clouds_groups.back().emplace_back(
                    std::move(cloud_current));
                }
            }

          clouds_groups.insert(clouds_groups.end(),
                               op_clouds_groups.begin(),
                               op_clouds_groups.end());
        }

      return clouds_groups;
    }

    // Find all clouds for a given order parameter
    std::vector<Cloud<dim>>
    find_clouds_for_order_parameter(const BlockVectorType &solution,
                                    const unsigned int     order_parameter_id)
    {
      std::vector<Cloud<dim>> clouds;

      // Loop through the whole mesh and set the user flags to false (so
      // everything is considered unmarked)
      for (auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              cell->clear_user_flag();
            }
        }

      clouds.emplace_back(order_parameter_id);

      // Vector to evaluate nodal values
      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      // The flood fill loop
      for (auto &cell : dof_handler.cell_iterators_on_level(0))
        {
          bool grain_assigned = false;
          recursive_flood_fill(cell,
                               solution,
                               order_parameter_id,
                               values,
                               clouds.back(),
                               grain_assigned);

          if (grain_assigned)
            {
              // Get a new cloud initialized for the next grain to be found
              clouds.emplace_back(order_parameter_id);
            }
        }

      // If the last grain was initialized but empty, delete it
      if (clouds.back().get_cells().size() == 0)
        {
          clouds.pop_back();
        }

      // Merge clouds from multiple processors
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        {
          // Get all clouds from other ranks
          auto global_clouds =
            Utilities::MPI::all_gather(MPI_COMM_WORLD, clouds);

          // Now gather all the clouds
          clouds.clear();
          for (auto &part_cloud : global_clouds)
            {
              std::move(part_cloud.begin(),
                        part_cloud.end(),
                        std::back_inserter(clouds));
            }

          /* When distributed across multiple processors, some clouds may be in
           * fact parts of a single one, we then merge the touching clouds into
           * one.
           */
          stitch_clouds(clouds);
        }

      return clouds;
    }

    // Recursive flood fill algorithm
    void
    recursive_flood_fill(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &cell,
      const BlockVectorType &                               solution,
      const unsigned int                                    order_parameter_id,
      Vector<Number> &                                      values,
      Cloud<dim> &                                          cloud,
      bool &                                                grain_assigned)
    {
      // If a cell has children, then we iterate over them
      if (cell->has_children())
        {
          for (unsigned int n = 0; n < cell->n_children(); n++)
            {
              recursive_flood_fill(cell->child(n),
                                   solution,
                                   order_parameter_id,
                                   values,
                                   cloud,
                                   grain_assigned);
            }
        }
      // If this is an active cell, the process it
      else if (!cell->user_flag_set() && cell->is_locally_owned())
        {
          cell->set_user_flag();

          // Get average value of the order parameter for the cell
          cell->get_dof_values(solution.block(order_parameter_id +
                                              order_parameters_offset),
                               values);

          double etai = std::accumulate(values.begin(),
                                        values.end(),
                                        0.0,
                                        std::plus<double>()) /
                        values.size();

          /* Check if the cell is inside the grain described by the given order
           * parameter. If so, add the current cell to the current cloud.
           */
          if (etai > threshold_lower && etai < threshold_upper)
            {
              grain_assigned = true;
              cloud.add_cell(*cell);

              // Check if this cell is at the interface with another rank
              for (const auto f : cell->face_indices())
                {
                  if (!cell->at_boundary(f) && cell->neighbor(f)->is_active() &&
                      cell->neighbor(f)->is_ghost())
                    {
                      cloud.add_edge_cell(*cell);
                      break;
                    }
                }

              bool is_periodic_primary = false;

              // Recursive call for all neighbors
              for (const auto f : cell->face_indices())
                {
                  if (!cell->at_boundary(f))
                    {
                      recursive_flood_fill(cell->neighbor(f),
                                           solution,
                                           order_parameter_id,
                                           values,
                                           cloud,
                                           grain_assigned);
                    }
                  else if (cell->has_periodic_neighbor(f))
                    {
                      cloud.add_periodic_secondary_cell(
                        *cell->periodic_neighbor(f));
                      is_periodic_primary = true;
                    }
                }

              // Check if cell is periodic primary
              if (is_periodic_primary)
                {
                  cloud.add_periodic_primary_cell(*cell);
                }
            }
        }
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
              if (gr_base.get_grain_id() != gr_other.get_grain_id())
                {
                  const bool are_neighbors =
                    force_reassignment ?
                      gr_base.get_order_parameter_id() ==
                        gr_other.get_order_parameter_id() :
                      gr_base.get_old_order_parameter_id() ==
                        gr_other.get_old_order_parameter_id();

                  if (are_neighbors)
                    {
                      gr_base.add_neighbor(&gr_other);
                    }
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

    // Merge different parts of clouds if they touch if MPI distributed
    void
    stitch_clouds(std::vector<Cloud<dim>> &clouds) const
    {
      for (unsigned int cl_primary = 0; cl_primary < clouds.size();
           ++cl_primary)
        {
          auto &cloud_primary = clouds[cl_primary];

          for (unsigned int cl_secondary = cl_primary + 1;
               cl_secondary < clouds.size();
               ++cl_secondary)
            {
              auto &cloud_secondary = clouds[cl_secondary];

              /* If two clouds touch each other, we then append all the cells of
               * the primary clouds to the cells of the secondary one and erase
               * the primary cell.
               */
              if (cloud_primary.is_stitchable_with(cloud_secondary))
                {
                  cloud_secondary.stitch(cloud_primary);
                  clouds.erase(clouds.begin() + cl_primary);
                  cl_primary--;
                  break;
                }
            }
        }
    }

    // Print clouds (mainly for debug)
    template <typename Stream>
    void
    print_clouds(const std::vector<Cloud<dim>> &clouds, Stream &out) const
    {
      unsigned int cloud_id = 0;

      out << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
          << " Number of clouds = " << clouds.size() << std::endl;
      for (auto &cloud : clouds)
        {
          Segment<dim> current_segment(cloud);

          out << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " "
              << "cloud_id = " << cloud_id
              << " | cloud order parameter = " << cloud.get_order_parameter_id()
              << " | center = " << current_segment.get_center()
              << " | radius = " << current_segment.get_radius()
              << " | number of cells = " << cloud.get_cells().size()
              << std::endl;
          cloud_id++;
        }
    }

    // Output clods (mainly for debug purposes)
    void
    output_clouds(const std::vector<Cloud<dim>> &clouds,
                  const bool                     is_merged) const
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = false;

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      // Identify all order parameters in use by the given clouds
      std::set<unsigned int> current_order_parameters;
      for (const auto &cl : clouds)
        {
          current_order_parameters.insert(cl.get_order_parameter_id());
        }

      // Total number of cells and order parameters
      const unsigned int n_cells =
        dof_handler.get_triangulation().n_active_cells();

      std::map<unsigned int, Vector<float>> order_parameter_indicators;

      // Initialize with invalid order parameter (negative)
      for (const auto &op : current_order_parameters)
        {
          order_parameter_indicators.emplace(op, n_cells);
          order_parameter_indicators.at(op) = -1.;
        }

      // For each order parameter identify cells contained in its clouds
      unsigned int counter = 0;
      for (auto &tria_cell :
           dof_handler.get_triangulation().active_cell_iterators())
        {
          for (const auto &cl : clouds)
            {
              for (const auto &cell : cl.get_cells())
                {
                  if (cell.barycenter().distance(tria_cell->barycenter()) <
                      1e-6)
                    {
                      order_parameter_indicators.at(
                        cl.get_order_parameter_id())[counter] =
                        cl.get_order_parameter_id();
                    }
                }
            }
          counter++;
        }

      // Build output
      data_out.attach_triangulation(dof_handler.get_triangulation());
      for (const auto &op : current_order_parameters)
        {
          data_out.add_data_vector(order_parameter_indicators.at(op),
                                   "op" + std::to_string(op));
        }
      data_out.build_patches();

      pcout << "Outputing clouds..." << std::endl;

      /* This function can be called for global clouds after they have been
       * indeitifed for each order parameter and populated to each rank or for
       * local clouds which exist locally only on a given rank. For local calls,
       * the order parameters may be different for each processor. When calling
       * write_vtu_in_parallel(), only those order parameter which present at
       * each processor enter the resultant vtu file. In general, that means
       * that for local calls the output file in general case will be empty. To
       * avoid this, we write separate outputs for each of the processor. They
       * are totally independent from each other and may contain different order
       * parameters, for this reason a pvtu record is not generated to merge
       * them all together.
       */
      if (is_merged)
        {
          static unsigned int counter_merged = 0;

          const std::string filename =
            "clouds_merged." + std::to_string(counter_merged) + ".vtu";
          data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

          counter_merged++;
        }
      else
        {
          static unsigned int counter_split = 0;

          const std::string filename =
            "clouds_split." + std::to_string(counter_split) + "." +
            std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
            ".vtu";
          std::ofstream output_stream(filename);
          data_out.write_vtu(output_stream);

          counter_split++;
        }
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

    // Perform greedy initialization
    const bool greedy_init;

    // Minimum value of order parameter value
    const double threshold_lower;

    // Maximum value of order parameter value
    const double threshold_upper;

    // Buffer zone around the grain
    const double buffer_distance_ratio;

    // Default order parameter id
    const unsigned int default_order_parameter_id;

    // Order parameters offset in FESystem
    const unsigned int order_parameters_offset;

    std::map<unsigned int, Grain<dim>> grains;
    std::map<unsigned int, Grain<dim>> old_grains;
    std::set<unsigned int>             active_order_parameters;

    static constexpr int max_order_parameters_num = MAX_SINTERING_GRAINS;

    // Last set of detected clouds
    std::vector<Cloud<dim>> last_clouds;

    ConditionalOStream pcout;
  };
} // namespace GrainTracker
