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

    bool
    track(const BlockVectorType &solution)
    {
      // TODO: dummy yet
      auto clouds = get_clouds(solution);

      auto old_grains = grains;

      (void)clouds;
      (void)old_grains;

      return true;
    }

    std::tuple<bool, bool>
    initial_setup(const BlockVectorType &solution)
    {
      // At this point we imply that we have 1 variable per grain, so we set up
      // indices accordinly
      auto clouds = get_clouds(solution);

      std::set<unsigned int> grains_indices;
      for (const auto &cl : clouds)
        {
          grains_indices.insert(grains_indices.end(), cl.get_grain_id());
        }

      // At the moment index correlates the variable number, should be decoupled
      for (auto id : grains_indices)
        {
          grains.emplace_back(id, default_op_id, id);
        }

      // Now add segments to each grain representation
      for (const auto &cl : clouds)
        {
          Segment<dim> segment(cl);
          grains[cl.get_grain_id()].add_segment(segment);
        }

      std::cout << "All assigned to default. Number of grains: "
                << grains.size() << std::endl;
      for (const auto &gr : grains)
        {
          std::cout << "op_index = " << gr.get_order_parameter_id()
                    << " | segments = " << gr.get_segments().size()
                    << " | grain_index = " << gr.get_grain_id() << std::endl;
          for (const auto &segment : gr.get_segments())
            {
              std::cout << "    SEG: center = " << segment.get_center()
                        << " | radius = " << segment.get_radius() << std::endl;
            }
        }

      // Initial grains reassignment
      bool prefer_closest        = true;
      bool has_reassigned_grains = reassign_grains(max_grains, prefer_closest);

      active_order_parameters = build_active_op_ids();
      bool has_op_number_changed =
        (active_order_parameters != build_old_op_ids());

      return std::make_tuple(has_reassigned_grains, has_op_number_changed);
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

      for (auto &gr : grains)
        {
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

  private:
    std::set<unsigned int>
    build_active_op_ids() const
    {
      std::set<unsigned int> active_op_ids;

      for (const auto &gr : grains)
        {
          active_op_ids.insert(active_op_ids.end(),
                               gr.get_order_parameter_id());
        }

      return active_op_ids;
    }

    std::set<unsigned int>
    build_old_op_ids() const
    {
      std::set<unsigned int> old_op_ids;

      for (const auto &gr : grains)
        {
          old_op_ids.insert(old_op_ids.end(), gr.get_old_order_parameter_id());
        }

      return old_op_ids;
    }

    std::vector<Cloud<dim>>
    get_clouds(const BlockVectorType &solution)
    {
      std::vector<Cloud<dim>> clouds;

      const unsigned int n_order_params = solution.n_blocks() - 2;

      for (unsigned int current_grain_id = 0; current_grain_id < n_order_params;
           current_grain_id++)
        {
          detect_clouds(solution, current_grain_id, clouds);
        }

      return clouds;
    }

    void
    detect_clouds(const BlockVectorType &  solution,
                  const unsigned int       order_parameter_id,
                  std::vector<Cloud<dim>> &clouds)
    {
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
      for (auto &cell : dof_handler.active_cell_iterators())
        {
          bool grain_assigned = false;
          recursive_flood_fill(
            cell, solution, order_parameter_id, values, clouds, grain_assigned);

          if (grain_assigned)
            {
              // Get the grain set initialized for the next grain to be found
              Cloud<dim> new_cloud;
              clouds.push_back(new_cloud);
            }
        }

      // If the last grain was initialized but empty, delete it
      if (clouds.back().get_cells().size() == 0)
        {
          clouds.pop_back();
        }
    }

    void
    recursive_flood_fill(
      const TriaIterator<DoFCellAccessor<dim, dim, false>> &cell,
      const BlockVectorType &                               solution,
      const unsigned int                                    order_parameter_id,
      Vector<Number> &                                      values,
      std::vector<Cloud<dim>> &                             clouds,
      bool &                                                grain_assigned)
    {
      // Check if the cell has been marked already
      if (!cell->user_flag_set() && cell->is_locally_owned())
        {
          cell->set_user_flag();

          cell->get_dof_values(solution.block(order_parameter_id + 2), values);

          double etai = std::accumulate(values.begin(),
                                        values.end(),
                                        0,
                                        std::plus<double>()) /
                        values.size();

          if (etai > threshold_lower && etai < threshold_upper)
            {
              grain_assigned = true;
              clouds.back().add_cell(*cell);

              // Recursive call for all neighbors
              for (unsigned int n = 0; n < cell->n_faces(); n++)
                {
                  if (!cell->at_boundary(n))
                    {
                      recursive_flood_fill(cell->neighbor(n),
                                           solution,
                                           order_parameter_id,
                                           values,
                                           clouds,
                                           grain_assigned);
                    }
                }
            }
        }
    }

    bool
    reassign_grains(const unsigned int max_op_num, const bool prefer_closest)
    {
      bool grains_reassigned = true;

      for (int iter = max_op_num; iter >= 0; iter--)
        {
          for (unsigned int g_base_id = 0; g_base_id < grains.size();
               g_base_id++)
            {
              auto &gr_base = grains[g_base_id];

              const unsigned int op_base_id = gr_base.get_order_parameter_id();

              for (unsigned int g_other_id = 0; g_other_id < grains.size();
                   g_other_id++)
                {
                  if (g_other_id != g_base_id)
                    {
                      auto &gr_other = grains[g_other_id];

                      const unsigned int op_other_id =
                        gr_other.get_order_parameter_id();

                      double min_distance = gr_base.distance(gr_other);

                      if ((min_distance < 2.0 * buffer_distance) &&
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

                          for (unsigned int g_candidate_id = 0;
                               g_candidate_id < grains.size();
                               g_candidate_id++)
                            {
                              if (g_candidate_id != g_base_id)
                                {
                                  auto &gr_candidate = grains[g_candidate_id];

                                  unsigned int op_candidate_id =
                                    gr_candidate.get_order_parameter_id();

                                  double spacing =
                                    gr_base.distance(gr_candidate);

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
                            }

                          grains_reassigned = true;
                        }
                    }
                }
            }
        }

      // Build up neighbors connectivity
      for (auto &gr_base : grains)
        {
          for (auto &gr_other : grains)
            {
              gr_base.add_neighbor(&gr_other);
            }
        }

      return grains_reassigned;
    }

    const DoFHandler<dim> &dof_handler;

    const double       tol{1e-2};
    const double       threshold_lower{0 + tol};
    const double       threshold_upper{1 + tol};
    const double       buffer_distance{2.8};
    const unsigned int default_op_id{0};

    std::vector<Grain<dim>> grains;
    std::set<unsigned int>  active_order_parameters;

    static constexpr int max_grains = MAX_SINTERING_GRAINS;
  };
} // namespace GrainTracker
