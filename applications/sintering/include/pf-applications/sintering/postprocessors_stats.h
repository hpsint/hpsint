// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/table_handler.h>

#include <pf-applications/sintering/operator_sintering_coupled_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>
#include <pf-applications/sintering/postprocessors.h>

namespace Sintering
{
  namespace Postprocessors
  {
    using namespace hpsint;

    template <int dim, typename VectorType>
    void
    output_porosity_stats(
      const DoFHandler<dim>                        &dof_handler,
      const VectorType                             &solution,
      const std::string                             output,
      const double                                  threshold_upper = 0.8,
      std::shared_ptr<const BoundingBoxFilter<dim>> box_filter      = nullptr)
    {
      const double invalid_particle_id = -1.0; // TODO

      const auto comm = dof_handler.get_mpi_communicator();

      // Detect pores and assign ids
      auto [particle_ids, local_to_global_particle_ids, offset] =
        internal::detect_pores(dof_handler,
                               solution,
                               invalid_particle_id,
                               threshold_upper,
                               box_filter);

      const unsigned int n_pores =
        GrainTracker::number_of_stitched_particles(local_to_global_particle_ids,
                                                   comm);

      GrainTracker::switch_to_global_indices(particle_ids,
                                             local_to_global_particle_ids,
                                             offset,
                                             invalid_particle_id);

      const auto [pores_centers, pores_measures] =
        GrainTracker::compute_particles_info(dof_handler,
                                             particle_ids,
                                             n_pores,
                                             invalid_particle_id);

      const auto [pores_radii, pores_remotes] =
        GrainTracker::compute_particles_radii(
          dof_handler, particle_ids, pores_centers, false, invalid_particle_id);
      (void)pores_remotes;

      if (Utilities::MPI::this_mpi_process(comm) != 0)
        return;

      TableHandler table;

      std::vector<std::string> labels{"x", "y", "z"};

      for (unsigned int i = 0; i < n_pores; ++i)
        {
          table.add_value("id", i);
          table.add_value("measure", pores_measures[i]);
          table.add_value("radius", pores_radii[i]);

          for (unsigned int d = 0; d < dim; ++d)
            table.add_value(labels[d], pores_centers[i][d]);
        }

      // Output to file
      std::stringstream ss;
      table.write_text(ss);

      std::ofstream out_file(output);
      out_file << ss.rdbuf();
      out_file.close();
    }

    template <int dim,
              typename NonLinearOperator,
              typename VectorType,
              typename VectorizedArrayType,
              typename Number>
    void
    output_grains_stats(
      const Mapping<dim>                       &mapping,
      const DoFHandler<dim>                    &dof_handler,
      const NonLinearOperator                  &sintering_operator,
      const GrainTracker::Tracker<dim, Number> &grain_tracker,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
                       &advection_mechanism,
      const VectorType &solution,
      const std::string output)
    {
      const auto comm = dof_handler.get_mpi_communicator();

      const bool has_ghost_elements = solution.has_ghost_elements();

      if (has_ghost_elements == false)
        solution.update_ghost_values();

      // We assume that a grain contains a single segment
      constexpr unsigned int segment_id = 0;

      Tensor<1, dim, Number> dummy_velocities;
      for (unsigned int d = 0; d < dim; ++d)
        dummy_velocities[d] = std::numeric_limits<double>::quiet_NaN();

      // Two different ways to compute velocities
      std::function<Tensor<1, dim, Number>(const unsigned int)> get_velocity;
      if constexpr (std::is_base_of_v<
                      SinteringOperatorCoupledBase<dim,
                                                   Number,
                                                   VectorizedArrayType,
                                                   NonLinearOperator>,
                      NonLinearOperator>)
        {
          std::vector<Point<dim>> evaluation_points;

          std::array<std::vector<Number>, dim> displacements;

          std::map<unsigned int, unsigned int> grain_id_to_eval_id;
          unsigned int                         counter = 0;

          for (const auto &[grain_id, grain] : grain_tracker.get_grains())
            {
              grain_id_to_eval_id[grain_id] = counter++;
              evaluation_points.push_back(
                grain.get_segments()[segment_id].get_center());
            }

          // set up cache manually
          Utilities::MPI::RemotePointEvaluation<dim, dim> rpe;
          rpe.reinit(evaluation_points,
                     dof_handler.get_triangulation(),
                     mapping);

          // use the cache
          for (unsigned int b = solution.n_blocks() - dim, d = 0;
               b < solution.n_blocks();
               ++b, ++d)
            {
              displacements[d] =
                VectorTools::point_values<1>(rpe,
                                             dof_handler,
                                             solution.block(b));
            }

          get_velocity =
            [displacements       = std::move(displacements),
             grain_id_to_eval_id = std::move(grain_id_to_eval_id),
             dummy_velocities    = std::move(dummy_velocities),
             dt = sintering_operator.get_data().time_data.get_current_dt(),
             &grain_tracker](const unsigned int grain_id) {
              Tensor<1, dim, Number> vt = dummy_velocities;

              if (grain_tracker.get_grains().at(grain_id).n_segments() == 1 &&
                  dt != 0)
                for (unsigned int d = 0; d < dim; ++d)
                  vt[d] =
                    displacements[d][grain_id_to_eval_id.at(grain_id)] / dt;

              return vt;
            };
        }
      else
        {
          (void)mapping;

          get_velocity = [dummy_velocities = std::move(dummy_velocities),
                          &advection_mechanism,
                          &grain_tracker](const unsigned int grain_id) {
            const Tensor<1, dim, Number> vt =
              (!advection_mechanism.get_grains_data().empty() &&
               grain_tracker.get_grains().at(grain_id).n_segments() == 1) ?
                advection_mechanism.get_translation_velocity_for_grain(
                  grain_tracker.get_grain_segment_index(grain_id, segment_id)) :
                dummy_velocities;

            return vt;
          };
        }

      if (has_ghost_elements == false)
        solution.zero_out_ghost_values();

      if (Utilities::MPI::this_mpi_process(comm) != 0)
        return;

      TableHandler table;

      const std::vector labels_coords{"x", "y", "z"};
      const std::vector labels_forces{"fx", "fy", "fz"};
      const std::vector labels_torques =
        (moment_s<dim, VectorizedArrayType> == 1) ?
          std::vector({"t"}) :
          std::vector({"tx", "ty", "tz"});
      const std::vector labels_velocities{"vx", "vy", "vz"};

      const auto dummy =
        create_array<1 + dim + moment_s<dim, VectorizedArrayType>>(
          std::numeric_limits<double>::quiet_NaN());

      for (const auto &[grain_id, grain] : grain_tracker.get_grains())
        {
          table.add_value("id", grain_id);
          table.add_value("measure", grain.get_measure());
          table.add_value("radius", grain.get_max_radius());
          table.add_value("max_value", grain.get_max_value());
          table.add_value("order_parameter_id", grain.get_order_parameter_id());

          for (unsigned int d = 0; d < dim; ++d)
            table.add_value(labels_coords[d],
                            grain.n_segments() == 1 ?
                              grain.get_segments()[0].get_center()[d] :
                              std::numeric_limits<double>::quiet_NaN());

          if (advection_mechanism.enabled())
            {
              const Number *data =
                (!advection_mechanism.get_grains_data().empty() &&
                 grain.n_segments() == 1) ?
                  advection_mechanism.grain_data(
                    grain_tracker.get_grain_segment_index(grain_id, 0)) :
                  dummy.data();

              // Output volume should be less than measure
              table.add_value("volume", *data++);

              // Output forces
              for (unsigned int d = 0; d < dim; ++d)
                table.add_value(labels_forces[d], *data++);

              // Output torques
              for (unsigned int d = 0; d < moment_s<dim, VectorizedArrayType>;
                   ++d)
                table.add_value(labels_torques[d], *data++);

              // Output translation velocities
              const auto vt = get_velocity(grain_id);

              for (unsigned int d = 0; d < dim; ++d)
                table.add_value(labels_velocities[d], vt[d]);
            }
        }

      // Output to file
      std::stringstream ss;
      table.write_text(ss);

      std::ofstream out_file(output);
      out_file << ss.rdbuf();
      out_file.close();
    }
  } // namespace Postprocessors
} // namespace Sintering