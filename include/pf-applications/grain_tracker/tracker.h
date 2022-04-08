#pragma once

#include <deal.II/grid/grid_tools.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <functional>

#include "cloud.h"
#include "grain.h"
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
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    Tracker(const DoFHandler<dim> &dof_handler,
            const double           threshold_lower       = 0.01,
            const double           threshold_upper       = 1.01,
            const double           buffer_distance_ratio = 0.05,
            const unsigned int     default_op_id         = 0,
            const unsigned int     op_offset             = 2)
      : dof_handler(dof_handler)
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
      const auto clouds = find_clouds(solution);

      // Copy old grains
      const auto old_grains = grains;

      // Clear current grains
      grains.clear();

      // Create segments and transfer grain_id's for them
      for (auto &cloud : clouds)
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
                  double distance = current_segment.distance(segment);

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
      const bool prefer_closest = false;

      // Reassign grains and return the result
      return reassign_grains(max_grains, prefer_closest);
    }

    /* Initialization of grains at the very first step. The function returns a
     * tuple of bool variables which signify if any grains have been reassigned
     * and if the number of active order parameters has been changed.
     */
    std::tuple<bool, bool>
    initial_setup(const BlockVectorType &solution)
    {
      /* At this point we imply that we have 1 variable per grain, so we set up
       * grain indices accordinly. The main reason for that is that we may have
       * periodic boundary conditions defined and thus a certain grain can
       * contain multiple segments. In the current implementation this
       * assumption allows to capture and then track such geometry during
       * analysis. A better approach is to use information regarding periodicity
       * provided by deal.II.
       */
      // TODO: get rid of this assumption.
      auto clouds = find_clouds(solution);

      // Create grains from clouds and add segments to each grain
      for (const auto &cl : clouds)
        {
          unsigned int gid = cl.get_order_parameter_id();
          grains.try_emplace(gid, gid, default_order_parameter_id, gid);

          Segment<dim> segment(cl);
          grains.at(cl.get_order_parameter_id()).add_segment(segment);
        }

      /* Initial grains reassignment, the closest neighbors are allowed as we
       * want to minimize the number of order parameters in use.
       */
      const bool prefer_closest = true;

      // Reassign grains and return the result
      return reassign_grains(max_grains, prefer_closest);
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

          double transfer_buffer =
            std::max(0.0, grain.distance_to_nearest_neighbor() / 2.0);

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

      for (const auto &[gid, gr] : grains)
        {
          (void)gid;

          if (gr.get_order_parameter_id() != gr.get_old_order_parameter_id())
            {
              /* Transfer buffer is the extra zone around the grain within which
               * the order parameters are swapped. Its maximum size is the half
               * of the distance to the nearest neighbor.
               */
              const unsigned int op_id_dst = gr.get_order_parameter_id();
              const unsigned int op_id_src = gr.get_old_order_parameter_id();

              std::ostringstream ss;
              ss << "Remap order parameter for grain id = " << gr.get_grain_id()
                 << ": from " << op_id_src << " to " << op_id_dst << std::endl;
              log.emplace_back(ss.str());

              /* At first we transfer the values from the dofs related to the
               * old order parameters to the dofs of the new order parameter.
               */
              alter_dof_values_for_grain(
                gr,
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
                gr,
                [this, &values, op_id_src](
                  const dealii::DoFCellAccessor<dim, dim, false> &cell,
                  BlockVectorType *                               solution) {
                  cell.get_dof_values(solution->block(op_id_src +
                                                      order_parameters_offset),
                                      values);
                  values = 0;
                  cell.set_dof_values(values,
                                      solution->block(op_id_src +
                                                      order_parameters_offset));
                });
            }
        }

      print_log(log);
    }

    // Get active order parameters ids
    std::set<unsigned int>
    get_active_order_parameters() const
    {
      return active_order_parameters;
    }

    // Print current grains
    template <typename Stream>
    void
    print_grains(Stream &out) const
    {
      out << "Number of grains: " << grains.size() << std::endl;
      for (const auto &[gid, gr] : grains)
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

    // Output last set of clouds
    void
    dump_last_clouds() const
    {
      print_clouds(last_clouds, pcout);
      output_clouds(last_clouds, /*is_merged = */ true);
    }

  private:
    // Build a set of active order parameters
    std::set<unsigned int>
    build_active_order_parameter_ids() const
    {
      std::set<unsigned int> active_op_ids;

      for (const auto &[gid, gr] : grains)
        {
          (void)gid;
          active_op_ids.insert(gr.get_order_parameter_id());
        }

      return active_op_ids;
    }

    // Build a set of old order parameters
    std::set<unsigned int>
    build_old_order_parameter_ids() const
    {
      std::set<unsigned int> old_op_ids;

      for (const auto &[gid, gr] : grains)
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
      last_clouds.clear();

      const unsigned int n_order_params = solution.n_blocks() - 2;

      for (unsigned int current_grain_id = 0; current_grain_id < n_order_params;
           ++current_grain_id)
        {
          auto op_clouds =
            find_clouds_for_order_parameter(solution, current_grain_id);
          last_clouds.insert(last_clouds.end(),
                             op_clouds.begin(),
                             op_clouds.end());
        }

      return last_clouds;
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
          merge_clouds(clouds);
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

              // Recursive call for all neighbors
              for (unsigned int n = 0; n < cell->n_faces(); n++)
                {
                  if (!cell->at_boundary(n))
                    {
                      recursive_flood_fill(cell->neighbor(n),
                                           solution,
                                           order_parameter_id,
                                           values,
                                           cloud,
                                           grain_assigned);
                    }
                }
            }
        }
    }

    // Reassign grains order parameters to prevent collision
    std::tuple<bool, bool>
    reassign_grains(const unsigned int max_order_parameters_num,
                    const bool         prefer_closest)
    {
      bool grains_reassigned = false;

      std::vector<std::string> log;

      /* Maximum number of reassignment iterations is equal to the maximum
       * number of order parameters available
       */
      for (unsigned int iter = 0; iter <= max_order_parameters_num; iter++)
        {
          // Base grain to compare with
          for (auto &[g_base_id, gr_base] : grains)
            {
              const unsigned int op_base_id = gr_base.get_order_parameter_id();

              // Secondary grain
              for (auto &[g_other_id, gr_other] : grains)
                {
                  if (g_other_id != g_base_id)
                    {
                      const unsigned int op_other_id =
                        gr_other.get_order_parameter_id();

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
                      if ((min_distance <
                           buffer_distance_base + buffer_distance_other) &&
                          (op_other_id == op_base_id))
                        {
                          std::ostringstream ss;
                          ss << "Found overlap between grain "
                             << gr_base.get_grain_id() << " and grain "
                             << gr_other.get_grain_id()
                             << " with order parameter " << op_base_id
                             << std::endl;

                          log.emplace_back(ss.str());

                          std::vector<Number> minimum_distance_list(
                            max_order_parameters_num,
                            std::numeric_limits<Number>::quiet_NaN());

                          /* Find a candidate op for gr_other grain. To this end
                           * we measure distance from the secondary grain to the
                           * other grains except for the current base one.
                           */
                          for (const auto &[g_candidate_id, gr_candidate] :
                               grains)
                            {
                              if (g_candidate_id != g_base_id &&
                                  g_candidate_id != g_other_id)
                                {
                                  const unsigned int op_candidate_id =
                                    gr_candidate.get_order_parameter_id();

                                  const double spacing =
                                    gr_other.distance(gr_candidate);

                                  if (std::isnan(minimum_distance_list
                                                   [op_candidate_id]) ||
                                      spacing <
                                        minimum_distance_list[op_candidate_id])
                                    {
                                      minimum_distance_list[op_candidate_id] =
                                        spacing;
                                    }
                                }
                            }

                          unsigned int new_op_index = op_other_id;

                          /* Now strategies for the order parameters
                           * reassignment are available: either we pick the
                           * closest order parameter that does not overlap with
                           * the secondary grain or we pick the most distant
                           * one.
                           */

                          /* The first option works for the very first step to
                           * minimize the total number of order parameters in
                           * the system.
                           */
                          if (prefer_closest)
                            {
                              for (unsigned int op = 0;
                                   op < minimum_distance_list.size();
                                   op++)
                                {
                                  if (op != op_other_id)
                                    {
                                      double current_distance =
                                        minimum_distance_list[op];

                                      if (!std::isnan(current_distance) &&
                                          minimum_distance_list[op] > 0)
                                        {
                                          new_op_index = op;
                                          break;
                                        }
                                    }
                                }
                            }

                          /* The second option is used for regular tracking, in
                           * this case we choose the most distant order
                           * parameter to reduce the number of future
                           * reassignments.
                           */
                          if (!prefer_closest || new_op_index == op_other_id)
                            {
                              double max_distance =
                                std::numeric_limits<double>::min();

                              for (unsigned int op = 0;
                                   op < minimum_distance_list.size();
                                   op++)
                                {
                                  if (op != op_other_id)
                                    {
                                      double current_distance =
                                        minimum_distance_list[op];

                                      if (!std::isnan(current_distance) &&
                                          minimum_distance_list[op] >
                                            max_distance)
                                        {
                                          max_distance =
                                            minimum_distance_list[op];
                                          new_op_index = op;
                                        }
                                      else if (std::isnan(current_distance))
                                        {
                                          new_op_index = op;
                                          break;
                                        }
                                    }
                                }
                            }

                          if (new_op_index != op_other_id)
                            {
                              gr_other.set_order_parameter_id(new_op_index);
                              grains_reassigned = true;
                            }
                        }
                    }
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
                    prefer_closest ? gr_base.get_order_parameter_id() ==
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

      active_order_parameters = build_active_order_parameter_ids();
      bool op_number_changed =
        (active_order_parameters != build_old_order_parameter_ids());

      print_log(log);

      return std::make_tuple(grains_reassigned, op_number_changed);
    }

    // Merge different parts of clouds if they touch if MPI distributed
    void
    merge_clouds(std::vector<Cloud<dim>> &clouds) const
    {
      for (unsigned int cl_primary = 0; cl_primary < clouds.size();
           ++cl_primary)
        {
          const auto &cloud_primary = clouds[cl_primary];

          for (unsigned int cl_secondary = cl_primary + 1;
               cl_secondary < clouds.size();
               ++cl_secondary)
            {
              auto &cloud_secondary = clouds[cl_secondary];

              bool do_stiching = false;

              for (const auto &cell_primary : cloud_primary.get_cells())
                {
                  for (const auto &cell_secondary : cloud_secondary.get_cells())
                    {
                      if (cell_primary.distance(cell_secondary) < 0.0)
                        {
                          do_stiching = true;
                          break;
                        }
                    }

                  if (do_stiching)
                    {
                      break;
                    }
                }

              /* If two clouds touch each other, we then append all the cells of
               * the primary clouds to the cells of the secondary one and erase
               * the primary cell.
               */
              if (do_stiching)
                {
                  for (const auto &cell_primary : cloud_primary.get_cells())
                    {
                      cloud_secondary.add_cell(cell_primary);
                    }
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
      std::set<unsigned int> active_order_parameters;
      for (const auto &cl : clouds)
        {
          active_order_parameters.insert(cl.get_order_parameter_id());
        }

      // Total number of cells and order parameters
      const unsigned int n_cells =
        dof_handler.get_triangulation().n_active_cells();

      std::map<unsigned int, Vector<float>> order_parameter_indicators;

      // Initialize with invalid order parameter (negative)
      for (const auto &op : active_order_parameters)
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
      for (const auto &op : active_order_parameters)
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
    std::set<unsigned int>             active_order_parameters;

    static constexpr int max_grains = MAX_SINTERING_GRAINS;

    // Last set of detected clouds
    std::vector<Cloud<dim>> last_clouds;

    ConditionalOStream pcout;
  };
} // namespace GrainTracker
