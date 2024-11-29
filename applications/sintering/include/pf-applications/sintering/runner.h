// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

//#define USE_FE_Q_iso_Q1

#ifdef USE_FE_Q_iso_Q1
#  define FE_DEGREE 2
#  define N_Q_POINTS_1D FE_DEGREE * 2
#else
#  define FE_DEGREE 1
#  define N_Q_POINTS_1D FE_DEGREE + 1
#endif

#define WITH_TIMING
//#define WITH_TIMING_OUTPUT
//#define DISABLE_MPI_IO_SURFACE_OUTPUT



#include <pf-applications/sintering/driver.h>
#include <pf-applications/sintering/initial_values_circle.h>
#include <pf-applications/sintering/initial_values_cloud.h>
#include <pf-applications/sintering/initial_values_debug.h>
#include <pf-applications/sintering/initial_values_hypercube.h>
#include <pf-applications/sintering/initial_values_microstructure_imaging.h>
#include <pf-applications/sintering/initial_values_microstructure_voronoi.h>

#include <cstdlib>
#include <regex>

using namespace dealii;

using Number              = double;
using VectorizedArrayType = VectorizedArray<Number>;

namespace Sintering
{
  namespace internal
  {
    void
    parse_params(const int              argc,
                 char **                argv,
                 const unsigned int     offset,
                 Sintering::Parameters &params,
                 ConditionalOStream &   pcout)
    {
      if (static_cast<unsigned int>(argc) >= offset + 1)
        {
          pcout << "Input parameters file:" << std::endl;
          pcout << std::ifstream(argv[offset]).rdbuf() << std::endl;

          params.parse(std::string(argv[offset]));
        }

      // Override params directly via command line
      for (unsigned int i = offset + 1; i < static_cast<unsigned int>(argc);
           ++i)
        {
          const std::string flag = std::string(argv[i]);

          if (flag.substr(0, 2) == "--")
            {
              std::regex  rgx("--([(\\w*).]*)=\"?(.*)\"?");
              std::smatch matches;

              AssertThrow(std::regex_search(flag, matches, rgx),
                          ExcMessage("Incorrect parameter string specified: " +
                                     flag + "\nThe correct format is:\n" +
                                     "--Path.To.Option=\"value\"\nor\n" +
                                     "--Path.To.Option=value\n" +
                                     "if 'value' does not contain spaces"));

              const std::string param_name  = matches[1].str();
              const std::string param_value = matches[2].str();

              params.set(param_name, param_value);
            }
        }

      params.check();

      pcout << "Parameters in JSON format:" << std::endl;
      params.print_input();
      pcout << std::endl;
    }
  } // namespace internal

  template <template <int dim, typename Number, typename VectorizedArrayType>
            typename NonLinearOperator,
            template <typename VectorizedArrayType>
            typename FreeEnergy>
  void
  runner(int argc, char **argv)
  {
    using SinteringProblem = Sintering::Problem<SINTERING_DIM,
                                                NonLinearOperator,
                                                FreeEnergy,
                                                Number,
                                                VectorizedArrayType>;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    Sintering::Parameters params;

    const std::string mode(argv[1]);

    if (argc == 1 || mode == "--help")
      {
        params.print_help();
      }
    else if (mode == "--circle")
      {
        AssertThrow(argc >= 4, ExcNotImplemented());

        // geometry
        const double       r0       = atof(argv[2]);
        const unsigned int n_grains = atoi(argv[3]);

        AssertThrow(r0 > 0,
                    ExcMessage("Particle radius should be grater than 0!"));
        AssertThrow(n_grains > 0,
                    ExcMessage("Number of grains should be grater than 0!"));

        // Output case specific info
        pcout << "Mode:             circle" << std::endl;
        pcout << "Grains radius:    " << r0 << std::endl;
        pcout << "Number of grains: " << n_grains << std::endl;
        pcout << std::endl;

        internal::parse_params(argc, argv, 4, params, pcout);

        const auto initial_solution =
          std::make_shared<Sintering::InitialValuesCircle<SINTERING_DIM>>(
            r0,
            params.geometry_data.interface_width,
            n_grains,
            params.geometry_data.minimize_order_parameters);

        AssertThrow(initial_solution->n_order_parameters() <=
                      MAX_SINTERING_GRAINS,
                    Sintering::ExcMaxGrainsExceeded(
                      initial_solution->n_order_parameters(),
                      MAX_SINTERING_GRAINS));

        SinteringProblem problem(params, initial_solution);
      }
    else if (mode == "--hypercube")
      {
        AssertThrow(argc >= 3 + SINTERING_DIM, ExcNotImplemented());

        // geometry
        const double r0 = atof(argv[2]);

        std::array<unsigned int, SINTERING_DIM> n_grains;
        for (unsigned int d = 0; d < SINTERING_DIM; ++d)
          n_grains[d] = atoi(argv[3 + d]);

        AssertThrow(r0 > 0,
                    ExcMessage("Particle radius should be grater than 0!"));
        AssertThrow(std::all_of(n_grains.begin(),
                                n_grains.end(),
                                [](const auto &val) { return val > 0; }),
                    ExcMessage("Number of grains should be grater than 0!"));

        const unsigned int n_total_grains = std::accumulate(
          n_grains.begin(), n_grains.end(), 1, std::multiplies<unsigned int>());

        // Output case specific info
        pcout << "Mode:             hypercube" << std::endl;
        pcout << "Grains radius:    " << r0 << std::endl;
        pcout << "Number of grains: ";
        for (unsigned int d = 0; d < SINTERING_DIM; ++d)
          {
            pcout << std::to_string(n_grains[d]);

            if (d + 1 != SINTERING_DIM)
              pcout << "x";
          }
        pcout << " = " << n_total_grains << std::endl;
        pcout << std::endl;

        internal::parse_params(argc, argv, 3 + SINTERING_DIM, params, pcout);

        // By default, 2 order parameters are enough for the packing description
        // If minimization is not required, then 0 value disables it
        const unsigned int n_order_params_to_use =
          params.geometry_data.minimize_order_parameters ? 2 : 0;

        const auto initial_solution =
          std::make_shared<Sintering::InitialValuesHypercube<SINTERING_DIM>>(
            r0,
            params.geometry_data.interface_width,
            n_grains,
            n_order_params_to_use);

        AssertThrow(initial_solution->n_order_parameters() <=
                      MAX_SINTERING_GRAINS,
                    Sintering::ExcMaxGrainsExceeded(
                      initial_solution->n_order_parameters(),
                      MAX_SINTERING_GRAINS));

        SinteringProblem problem(params, initial_solution);
      }
    else if (mode == "--cloud")
      {
        AssertThrow(argc >= 3,
                    ExcMessage("Argument cloud_file has to be provided!"));

        std::string   file_cloud = std::string(argv[2]);
        std::ifstream fstream(file_cloud);
        AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

        const auto particles =
          Sintering::read_particles<SINTERING_DIM>(fstream);

        // Output case specific info
        pcout << "Mode:       cloud" << std::endl;
        pcout << "Cloud path: " << file_cloud << std::endl;
        pcout << std::endl;

        pcout << "Particles list:" << std::endl;
        fstream.clear();
        fstream.seekg(0);
        pcout << fstream.rdbuf();
        pcout << std::endl;

        internal::parse_params(argc, argv, 3, params, pcout);

        pcout << "Parameters in JSON format:" << std::endl;
        params.print_input();
        pcout << std::endl;

        const auto initial_solution =
          std::make_shared<Sintering::InitialValuesCloud<SINTERING_DIM>>(
            particles,
            params.geometry_data.interface_width,
            params.geometry_data.minimize_order_parameters,
            params.geometry_data.interface_buffer_ratio,
            params.geometry_data.radius_buffer_ratio);

        AssertThrow(initial_solution->n_order_parameters() <=
                      MAX_SINTERING_GRAINS,
                    Sintering::ExcMaxGrainsExceeded(
                      initial_solution->n_order_parameters(),
                      MAX_SINTERING_GRAINS));

        SinteringProblem problem(params, initial_solution);
      }
    else if (mode == "--voronoi" || mode == "--imaging")
      {
        static_assert(SINTERING_DIM == 2,
                      "Only 2D case is currently supported");

        AssertThrow(argc >= 3,
                    ExcMessage("Argument file_name has to be provided!"));

        std::string   input_file = std::string(argv[2]);
        std::ifstream fstream(input_file);
        AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

        // Output case specific info
        pcout << "Mode:       " << mode.substr(2) << std::endl;
        pcout << "Input file: " << input_file << std::endl;
        pcout << std::endl;

        internal::parse_params(argc, argv, 3, params, pcout);

        pcout << "Parameters in JSON format:" << std::endl;
        params.print_input();
        pcout << std::endl;

        const InterfaceDirection interface_direction(
          to_interface_direction(params.geometry_data.interface_direction));

        std::shared_ptr<InitialValues<SINTERING_DIM>> initial_solution;

        if (mode == "--voronoi")
          initial_solution =
            std::make_shared<Sintering::InitialValuesMicrostructureVoronoi>(
              fstream,
              params.geometry_data.interface_width,
              interface_direction,
              FreeEnergy<VectorizedArrayType>::op_components_offset);
        else if (mode == "--imaging")
          initial_solution =
            std::make_shared<Sintering::InitialValuesMicrostructureImaging>(
              fstream,
              params.geometry_data.interface_width,
              interface_direction,
              FreeEnergy<VectorizedArrayType>::op_components_offset);
        else
          AssertThrow(false, ExcNotImplemented());

        pcout << "initial_n_particles  = " << initial_solution->n_particles()
              << std::endl;
        pcout << "initial_n_components = " << initial_solution->n_components()
              << std::endl;

        AssertThrow(initial_solution->n_order_parameters() <=
                      MAX_SINTERING_GRAINS,
                    Sintering::ExcMaxGrainsExceeded(
                      initial_solution->n_order_parameters(),
                      MAX_SINTERING_GRAINS));

        SinteringProblem problem(params, initial_solution);
      }
    else if (mode == "--restart")
      {
        AssertThrow(argc >= 3, ExcNotImplemented());

        const std::string restart_path = std::string(argv[2]);

        // Output case specific info
        pcout << "Mode:         restart" << std::endl;
        pcout << "Restart path: " << restart_path << std::endl;
        pcout << std::endl;

        internal::parse_params(argc, argv, 3, params, pcout);

        SinteringProblem problem(params, restart_path);
      }
    else if (mode == "--debug")
      {
        // Output case specific info
        pcout << "Mode: debug" << std::endl;

        internal::parse_params(argc, argv, 2, params, pcout);

        const auto initial_solution =
          std::make_shared<Sintering::InitialValuesDebug<SINTERING_DIM>>();

        SinteringProblem problem(params, initial_solution);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
  }
} // namespace Sintering
