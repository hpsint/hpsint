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

  template <int dim, typename Number>
  class Tracker
  {
  public:
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    Tracker(const DoFHandler<dim> &dof_handler)
      : dof_handler(dof_handler)
    {}

    std::tuple<bool, bool>
    track(const BlockVectorType &solution)
    {
      // Find cells clouds
      auto clouds = find_clouds(solution);

      // Copy old grains
      auto old_grains = grains;

      grains.clear();

      std::set<unsigned int> grains_indices;

      // Create segments and transfer grain_id's for them
      for (auto &cloud : clouds)
        {
          Segment<dim> current_segment(cloud);

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

          AssertThrow(
            grain_index_at_min_distance !=
              std::numeric_limits<unsigned int>::max(),
            ExcMessage(
              "Unable to detect a segment from the previous configuration for the cloud!"));

          // clang-format off
          AssertThrow(old_grains.at(grain_index_at_min_distance).get_order_parameter_id() 
            == cloud.get_order_parameter_id(),
            ExcMessage(
              std::string("Something got wrong with the order parameters numbering:\r\n") +
              std::string("    grain_index_at_min_distance = ") +
              std::to_string(grain_index_at_min_distance) + 
              std::string("    old grain order parameter   = ") +
              std::to_string(old_grains.at(grain_index_at_min_distance).get_order_parameter_id()) + 
              std::string("    cloud order parameter       = ") +
              std::to_string(cloud.get_order_parameter_id())
          ));
          // clang-format on

          auto insert_result =
            grains_indices.insert(grain_index_at_min_distance);
          if (insert_result.second)
            {
              grains.try_emplace(grain_index_at_min_distance,
                                 grain_index_at_min_distance,
                                 old_grains.at(grain_index_at_min_distance)
                                   .get_order_parameter_id(),
                                 old_grains.at(grain_index_at_min_distance)
                                   .get_order_parameter_id());
            }

          grains.at(grain_index_at_min_distance).add_segment(current_segment);
        }

      // Grains reassignment
      bool prefer_closest = false;

      return reassign_grains(max_grains, prefer_closest);
    }

    std::tuple<bool, bool>
    initial_setup(const BlockVectorType &solution)
    {
      // At this point we imply that we have 1 variable per grain, so we set up
      // indices accordinly
      auto clouds = find_clouds(solution);

      std::set<unsigned int> grains_indices;
      for (const auto &cl : clouds)
        {
          grains_indices.insert(cl.get_order_parameter_id());
        }

      for (auto gid : grains_indices)
        {
          grains.try_emplace(gid, gid, default_op_id, gid);
        }

      // Now add segments to each grain representation
      for (const auto &cl : clouds)
        {
          Segment<dim> segment(cl);
          grains.at(cl.get_order_parameter_id()).add_segment(segment);
        }

      // Initial grains reassignment
      bool prefer_closest = true;

      return reassign_grains(max_grains, prefer_closest);
    }

    void
    remap(BlockVectorType &solution)
    {
      remap({&solution});
    }

    void
    remap(std::vector<BlockVectorType *> solutions)
    {
      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      for (auto &[gid, gr] : grains)
        {
          (void)gid;

          if (gr.get_order_parameter_id() != gr.get_old_order_parameter_id())
            {
              double transfer_buffer =
                std::max(0.0, gr.distance_to_nearest_neighbor() / 2.0);

              const unsigned int op_id_dst = gr.get_order_parameter_id();
              const unsigned int op_id_src = gr.get_old_order_parameter_id();

              std::cout << "REMAP: from " << op_id_src << " to " << op_id_dst
                        << " | transfer_buffer = " << transfer_buffer
                        << std::endl;

              // At first we transfer the values
              for (auto &cell : dof_handler.active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      bool in_grain = false;
                      for (const auto &segment : gr.get_segments())
                        {
                          if (cell->barycenter().distance(
                                segment.get_center()) <
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
                              cell->get_dof_values(solution->block(op_id_src +
                                                                   2),
                                                   values);
                              cell->set_dof_values(values,
                                                   solution->block(op_id_dst +
                                                                   2));
                            }
                        }
                    }
                }

              for (auto &cell : dof_handler.active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      bool in_grain = false;
                      for (const auto &segment : gr.get_segments())
                        {
                          if (cell->barycenter().distance(
                                segment.get_center()) <
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
                              cell->get_dof_values(solution->block(op_id_src +
                                                                   2),
                                                   values);
                              values = 0;
                              cell->set_dof_values(values,
                                                   solution->block(op_id_src +
                                                                   2));
                            }
                        }
                    }
                }
            }
        }
    }

    std::set<unsigned int>
    get_active_order_parameters() const
    {
      return active_order_parameters;
    }

    template <typename Stream>
    void
    print_grains(Stream &out) const
    {
      out << "Number of grains: " << grains.size() << std::endl;
      for (const auto &[gid, gr] : grains)
        {
          (void)gid;
          out << "op_index = " << gr.get_order_parameter_id()
              << " | segments = " << gr.get_segments().size()
              << " | grain_index = " << gr.get_grain_id() << std::endl;
          for (const auto &segment : gr.get_segments())
            {
              out << "    segment: center = " << segment.get_center()
                  << " | radius = " << segment.get_radius() << std::endl;
            }
        }
      out << std::endl;
    }

  private:
    std::set<unsigned int>
    build_active_op_ids() const
    {
      std::set<unsigned int> active_op_ids;

      for (const auto &[gid, gr] : grains)
        {
          (void)gid;
          active_op_ids.insert(gr.get_order_parameter_id());
        }

      return active_op_ids;
    }

    std::set<unsigned int>
    build_old_op_ids() const
    {
      std::set<unsigned int> old_op_ids;

      for (const auto &[gid, gr] : grains)
        {
          (void)gid;
          old_op_ids.insert(gr.get_old_order_parameter_id());
        }

      return old_op_ids;
    }

    std::vector<Cloud<dim>>
    find_clouds(const BlockVectorType &solution)
    {
      std::vector<Cloud<dim>> clouds;

      const unsigned int n_order_params = solution.n_blocks() - 2;

      for (unsigned int current_grain_id = 0; current_grain_id < n_order_params;
           current_grain_id++)
        {
          auto op_clouds = find_clouds_for_op(solution, current_grain_id);
          clouds.insert(clouds.end(), op_clouds.begin(), op_clouds.end());
        }

      return clouds;
    }

    std::vector<Cloud<dim>>
    find_clouds_for_op(const BlockVectorType &solution,
                       const unsigned int     order_parameter_id)
    {
      std::vector<Cloud<dim>> clouds;

      // Loop through the whole mesh and set the user flags to false (so
      // everything is considered unmarked)
      for (auto &cell : dof_handler.active_cell_iterators())
        {
          cell->clear_user_flag();
        }

      Cloud<dim> cloud(order_parameter_id);
      clouds.push_back(cloud);

      // Vector to evaluate nodal values
      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      // The flood fill loop
      for (auto &cell : dof_handler.cell_iterators_on_level(0))
        // for (auto &cell : dof_handler.active_cell_iterators())
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
              Cloud<dim> new_cloud(order_parameter_id);
              clouds.push_back(new_cloud);
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

          clouds.clear();
          for (const auto &part_cloud : global_clouds)
            {
              clouds.insert(clouds.end(), part_cloud.begin(), part_cloud.end());
            }

          // Merge grains that are split across processors
          merge_clouds(clouds);
        }

      return clouds;
    }

    void
    recursive_flood_fill(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &cell,
      const BlockVectorType &                               solution,
      const unsigned int                                    order_parameter_id,
      Vector<Number> &                                      values,
      Cloud<dim> &                                          cloud,
      bool &                                                grain_assigned)
    {
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
      else if (!cell->user_flag_set() && cell->is_locally_owned())
        {
          cell->set_user_flag();

          cell->get_dof_values(solution.block(order_parameter_id + 2), values);

          double etai = std::accumulate(values.begin(),
                                        values.end(),
                                        0.0,
                                        std::plus<double>()) /
                        values.size();

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

    std::tuple<bool, bool>
    reassign_grains(const unsigned int max_op_num, const bool prefer_closest)
    {
      bool grains_reassigned = false;

      for (int iter = max_op_num; iter >= 0; iter--)
        {
          for (auto &[g_base_id, gr_base] : grains)
            {
              const unsigned int op_base_id = gr_base.get_order_parameter_id();

              for (auto &[g_other_id, gr_other] : grains)
                {
                  if (g_other_id != g_base_id)
                    {
                      const unsigned int op_other_id =
                        gr_other.get_order_parameter_id();

                      double min_distance = gr_base.distance(gr_other);
                      double buffer_distance_base =
                        buffer_distance_ratio * gr_base.get_max_radius();
                      double buffer_distance_other =
                        buffer_distance_ratio * gr_other.get_max_radius();

                      if ((min_distance <
                           buffer_distance_base + buffer_distance_other) &&
                          (op_other_id == op_base_id))
                        {
                          if (dealii::Utilities::MPI::this_mpi_process(
                                MPI_COMM_WORLD) == 0)
                            {
                              std::cout
                                << "Found overlap between grain "
                                << gr_base.get_grain_id() << " and grain "
                                << gr_other.get_grain_id()
                                << " with order parameter " << op_base_id
                                << std::endl;
                            }

                          std::vector<Number> minimum_distance_list(
                            max_op_num,
                            std::numeric_limits<Number>::quiet_NaN());

                          // Find a candidate op for gr_other grain
                          for (const auto &[g_candidate_id, gr_candidate] :
                               grains)
                            {
                              if (g_candidate_id != g_base_id &&
                                  g_candidate_id != g_other_id)
                                {
                                  unsigned int op_candidate_id =
                                    gr_candidate.get_order_parameter_id();

                                  double spacing =
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

      // Build up neighbors connectivity
      for (auto &[g_base_id, gr_base] : grains)
        {
          (void)g_base_id;
          for (const auto &[g_other_id, gr_other] : grains)
            {
              (void)g_other_id;
              gr_base.add_neighbor(&gr_other);
            }
        }

      active_order_parameters = build_active_op_ids();
      bool op_number_changed  = (active_order_parameters != build_old_op_ids());

      return std::make_tuple(grains_reassigned, op_number_changed);
    }

    void
    merge_clouds(std::vector<Cloud<dim>> &clouds) const
    {
      for (unsigned int cl_primary = 0; cl_primary < clouds.size();
           cl_primary++)
        {
          const auto &cloud_primary = clouds[cl_primary];

          for (unsigned int cl_secondary = cl_primary + 1;
               cl_secondary < clouds.size();
               cl_secondary++)
            {
              auto &cloud_secondary = clouds[cl_secondary];

              bool do_stiching = false;

              for (const auto &cell_primary : cloud_primary.get_cells())
                {
                  for (const auto &cell_secondary : cloud_secondary.get_cells())
                    {
                      if (cell_primary.distance(cell_secondary) < 0)
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

    const DoFHandler<dim> &dof_handler;

    const double       tol{1e-2};
    const double       threshold_lower{0 + tol};
    const double       threshold_upper{1 + tol};
    const double       buffer_distance_ratio{0.05};
    const unsigned int default_op_id{0};

    std::map<unsigned int, Grain<dim>> grains;
    std::set<unsigned int>             active_order_parameters;

    static constexpr int max_grains = MAX_SINTERING_GRAINS;
  };
} // namespace GrainTracker
