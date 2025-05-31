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

#include <pf-applications/sintering/parameters.h>
#include <pf-applications/sintering/postprocessors.h>
#include <pf-applications/sintering/projection.h>

#include <pf-applications/grid/bounding_box_filter.h>

namespace Sintering
{
  namespace Postprocessors
  {
    using namespace dealii;

    template <int dim,
              typename Number,
              typename VectorType,
              typename StreamType>
    void
    advanced_output(
      const MappingQ<dim>                               &mapping,
      const DoFHandler<dim>                             &dof_handler,
      const VectorType                                  &solution,
      const unsigned int                                 n_op,
      const OutputData                                  &params,
      const GrainTracker::Tracker<dim, Number>          &grain_tracker,
      const std::vector<ProjectedData<dim - 1, Number>> &sections,
      const std::vector<std::shared_ptr<const BoundingBoxFilter<dim>>>
                    &box_filters,
      TableHandler  &table,
      const double   t,
      MyTimerOutput &timer,
      const std::function<std::string(const std::string &, const unsigned int)>
                        &generate_name,
      const unsigned int counter,
      const std::string  label,
      StreamType        &pcout)
    {
      ScopedName sc("advanced_output");
      MyScope    scope(timer, sc);

      // Some settings
      const double iso_value = 0.5;

      if (params.table)
        {
          for (unsigned int i = 0; i < box_filters.size(); ++i)
            {
              if (params.iso_surf_area)
                {
                  const auto surface_area =
                    Postprocessors::compute_surface_area(
                      mapping,
                      dof_handler,
                      solution,
                      iso_value,
                      box_filters[i],
                      params.n_mca_subdivisions);

                  table.add_value(generate_name("iso_surf_area", i),
                                  surface_area);
                }

              if (params.iso_gb_area)
                {
                  const auto gb_area =
                    Postprocessors::compute_grain_boundaries_area(
                      mapping,
                      dof_handler,
                      solution,
                      iso_value,
                      n_op,
                      params.gb_threshold,
                      box_filters[i],
                      params.n_mca_subdivisions);

                  table.add_value(generate_name("iso_gb_area", i), gb_area);
                }

              if (params.coordination_number && !grain_tracker.empty())
                {
                  const auto avg_coord_num =
                    Postprocessors::compute_average_coordination_number(
                      dof_handler, n_op, grain_tracker, box_filters[i]);

                  table.add_value(generate_name("avg_coord_num", i),
                                  avg_coord_num);
                }
            }
        }

      for (unsigned int i = 0; i < box_filters.size(); ++i)
        {
          const std::string label_box = generate_name(label, i);

          if (params.contours)
            {
              const std::string output = params.vtk_path + "/contour_" +
                                         label_box + "." +
                                         std::to_string(counter) + ".vtu";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              Postprocessors::output_grain_contours_vtu(
                mapping,
                dof_handler,
                solution,
                iso_value,
                output,
                n_op,
                grain_tracker,
                params.n_coarsening_steps,
                box_filters[i],
                params.n_mca_subdivisions);

              if constexpr (dim == 3)
                for (unsigned int i = 0; i < sections.size(); ++i)
                  {
                    std::stringstream ss;
                    ss << params.vtk_path << "/contour_section_"
                       << params.sections[i].first << "="
                       << params.sections[i].second << "_" << label << "."
                       << counter << ".vtu";

                    const std::string output = ss.str();

                    pcout << "Outputing data at t = " << t << " (" << output
                          << ")" << std::endl;

                    Postprocessors::output_grain_contours_projected_vtu(
                      sections[i].state.mapping,
                      sections[i].state.dof_handler,
                      Postprocessors::BlockVectorWrapper<
                        std::vector<Vector<Number>>>(
                        sections[i].state.solution),
                      iso_value,
                      output,
                      n_op,
                      grain_tracker,
                      MPI_COMM_WORLD,
                      params.n_mca_subdivisions);
                  }
            }

          if (params.concentration_contour)
            {
              const std::string output = params.vtk_path + "/surface_" +
                                         label_box + "." +
                                         std::to_string(counter) + ".vtu";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              const auto only_concentration = solution.create_view(0, 1);

              Postprocessors::output_concentration_contour_vtu(
                mapping,
                dof_handler,
                *only_concentration,
                iso_value,
                output,
                params.n_coarsening_steps,
                box_filters[i],
                params.n_mca_subdivisions);
            }

          if (params.grain_boundaries)
            {
              const std::string output = params.vtk_path + "/gb_" + label_box +
                                         "." + std::to_string(counter) + ".vtu";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              Postprocessors::output_grain_boundaries_vtu(
                mapping,
                dof_handler,
                solution,
                iso_value,
                output,
                n_op,
                params.gb_threshold,
                params.n_coarsening_steps,
                box_filters[i],
                params.n_mca_subdivisions);
            }

          if (params.porosity)
            {
              const std::string output = params.vtk_path + "/porosity_" +
                                         label_box + "." +
                                         std::to_string(counter) + ".vtu";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              Postprocessors::output_porosity(mapping,
                                              dof_handler,
                                              solution,
                                              output,
                                              params.porosity_max_value,
                                              box_filters[i]);
            }

          if (params.porosity_stats)
            {
              const std::string output = params.vtk_path + "/porosity_stats_" +
                                         label_box + "." +
                                         std::to_string(counter) + ".log";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              Postprocessors::output_porosity_stats(dof_handler,
                                                    solution,
                                                    output,
                                                    params.porosity_max_value,
                                                    box_filters[i]);
            }

          if (params.porosity_contours)
            {
              const std::string output = params.vtk_path +
                                         "/porosity_contours_" + label_box +
                                         "." + std::to_string(counter) + ".vtu";

              pcout << "Outputing data at t = " << t << " (" << output << ")"
                    << std::endl;

              const auto only_concentration = solution.create_view(0, 1);

              Postprocessors::output_porosity_contours_vtu(
                mapping,
                dof_handler,
                *only_concentration,
                iso_value,
                output,
                params.n_coarsening_steps,
                box_filters[i],
                params.n_mca_subdivisions,
                params.porosity_smooth);
            }
        }

      if (params.contours_tex && !grain_tracker.empty())
        {
          const std::string output = params.vtk_path + "/contour_" + label +
                                     "." + std::to_string(counter) + ".txt";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          Postprocessors::output_grain_contours(mapping,
                                                dof_handler,
                                                solution,
                                                iso_value,
                                                output,
                                                n_op,
                                                grain_tracker);

          // Output sections for 3D case
          if constexpr (dim == 3)
            for (unsigned int i = 0; i < sections.size(); ++i)
              {
                std::stringstream ss;
                ss << params.vtk_path << "/contour_section_"
                   << params.sections[i].first << "="
                   << params.sections[i].second << "_" << label << "."
                   << counter << ".txt";

                const std::string output = ss.str();

                pcout << "Outputing data at t = " << t << " (" << output << ")"
                      << std::endl;

                Postprocessors::output_grain_contours_projected(
                  sections[i].state.mapping,
                  sections[i].state.dof_handler,
                  Postprocessors::BlockVectorWrapper<
                    std::vector<Vector<Number>>>(sections[i].state.solution),
                  iso_value,
                  output,
                  n_op,
                  grain_tracker,
                  sections[i].origin,
                  sections[i].normal,
                  MPI_COMM_WORLD);
              }
        }
    }
  } // namespace Postprocessors
} // namespace Sintering