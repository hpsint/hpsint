#pragma once

#include <deal.II/base/parameter_handler.h>

#include <pf-applications/sintering/preconditioners.h>

namespace Sintering
{
  struct ApproximationData
  {
    unsigned int fe_degree   = 1;
    unsigned int n_points_1D = 2;
  };

  struct GeometryData
  {
    unsigned int elements_per_interface    = 8; // 4 - works well with AMR=off
    double       boundary_factor           = 0.5;
    double       interface_width           = 2.0;
    bool         minimize_order_parameters = true;
  };

  struct AdaptivityData
  {
    double       top_fraction_of_cells    = 0.3;
    double       bottom_fraction_of_cells = 0.1;
    unsigned int min_refinement_depth     = 3;
    unsigned int max_refinement_depth     = 0;
    unsigned int refinement_frequency     = 10; // 0 - no refinement
  };

  struct GrainTrackerData
  {
    double       threshold_lower         = 0.01;
    double       threshold_upper         = 1.01;
    double       buffer_distance_ratio   = 0.05;
    unsigned int grain_tracker_frequency = 10; // 0 - no grain tracker
  };

  struct EnergyData
  {
    double A       = 16;
    double B       = 1;
    double kappa_c = 1;
    double kappa_p = 0.5;
  };

  struct MobilityData
  {
    double Mvol  = 1e-2;
    double Mvap  = 1e-10;
    double Msurf = 4;
    double Mgb   = 0.4;
    double L     = 1;
  };

  struct TimeIntegrationData
  {
    double       time_start                  = 0;
    double       time_end                    = 1e3;
    double       time_step_init              = 1e-3;
    double       time_step_min               = 1e-5;
    double       time_step_max               = 1e2;
    double       growth_factor               = 1.2;
    double       output_time_interval        = 10;
    unsigned int desirable_newton_iterations = 5;
    unsigned int desirable_linear_iterations = 100;
  };

  struct PreconditionersData
  {
    std::string outer_preconditioner = "BlockPreconditioner2";
    // std::string outer_preconditioner = "BlockPreconditioner3CH";
    // std::string outer_preconditioner = "ILU";

    BlockPreconditioner2Data   block_preconditioner_2_data;
    BlockPreconditioner3Data   block_preconditioner_3_data;
    BlockPreconditioner3CHData block_preconditioner_3_ch_data;
  };


  struct Parameters
  {
    ApproximationData   approximation_data;
    GeometryData        geometry_data;
    AdaptivityData      adaptivity_data;
    GrainTrackerData    grain_tracker_data;
    EnergyData          energy_data;
    MobilityData        mobility_data;
    TimeIntegrationData time_integration_data;
    PreconditionersData preconditioners_data;

    bool matrix_based = false;

    bool print_time_loop = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      prm.parse_input(file_name, "", true);

#ifdef FE_DEGREE
      AssertDimension(FE_DEGREE, approximation_data.fe_degree);
#endif

#ifdef N_Q_POINTS_1D
      AssertDimension(N_Q_POINTS_1D, approximation_data.n_points_1D);
#endif
    }

    void
    print()
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      ConditionalOStream pcout(
        std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      if (pcout.is_active())
        prm.print_parameters(
          pcout.get_stream(),
          ParameterHandler::OutputStyle::Description |
            ParameterHandler::OutputStyle::KeepDeclarationOrder);
    }

  private:
    void
    add_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Approximation");
      prm.add_parameter("FEDegree",
                        approximation_data.fe_degree,
                        "Degree of the shape the finite element.");
      prm.add_parameter("NPoints1D",
                        approximation_data.n_points_1D,
                        "Number of quadrature points.");
      prm.leave_subsection();


      prm.enter_subsection("Geometry");
      prm.add_parameter("ElementsPerInterface",
                        geometry_data.elements_per_interface,
                        "Number of elements per interface.");
      prm.add_parameter("BoundaryFactor",
                        geometry_data.boundary_factor,
                        "Bounding box padding ratio (to the largest radius).");
      prm.add_parameter("InterfaceWidth",
                        geometry_data.interface_width,
                        "Phase-field interface width.");
      prm.add_parameter("MinimizeOrderParameters",
                        geometry_data.minimize_order_parameters,
                        "Minimize number of initial order parameters.");
      prm.leave_subsection();


      prm.enter_subsection("Adaptivity");
      prm.add_parameter("TopFractionOfCells",
                        adaptivity_data.top_fraction_of_cells,
                        "Top fraction of cells.");
      prm.add_parameter("BottomFractionOfCells",
                        adaptivity_data.bottom_fraction_of_cells,
                        "Bottom fraction of cells.");
      prm.add_parameter("MinRefinementDepth",
                        adaptivity_data.min_refinement_depth,
                        "Minimum refinement depth.");
      prm.add_parameter("MaxRefinementDepth",
                        adaptivity_data.max_refinement_depth,
                        "Maximum refinement depth.");
      prm.add_parameter("RefinementFrequency",
                        adaptivity_data.refinement_frequency,
                        "Refinement frequency (0 = no refinement).");
      prm.leave_subsection();


      prm.enter_subsection("GrainTracker");
      prm.add_parameter("ThresholdLower",
                        grain_tracker_data.threshold_lower,
                        "Lower boundary for detecting grain.");
      prm.add_parameter("ThresholdUpper",
                        grain_tracker_data.threshold_upper,
                        "Upper boundary for detecting grain.");
      prm.add_parameter("BufferDistanceRatio",
                        grain_tracker_data.buffer_distance_ratio,
                        "Ratio of the transfer buffer (to the grain radius).");
      prm.add_parameter("GrainTrackerFrequency",
                        grain_tracker_data.grain_tracker_frequency,
                        "Grain tracker frequency (0 = no grain tracking).");
      prm.leave_subsection();


      prm.enter_subsection("Energy");
      prm.add_parameter("A", energy_data.A, "Energy parameter A.");
      prm.add_parameter("B", energy_data.B, "Energy parameter B.");
      prm.add_parameter("KappaC",
                        energy_data.kappa_c,
                        "Barrier height kappa_c.");
      prm.add_parameter("KappaP",
                        energy_data.kappa_p,
                        "Barrier height kappa_p.");
      prm.leave_subsection();


      prm.enter_subsection("Mobility");
      prm.add_parameter("Mvol",
                        mobility_data.Mvol,
                        "Volumetric diffusion mobility.");
      prm.add_parameter("Mvap",
                        mobility_data.Mvap,
                        "Vaporization diffusion mobility.");
      prm.add_parameter("Msurf",
                        mobility_data.Msurf,
                        "Surface diffusion  mobility.");
      prm.add_parameter("Mgb",
                        mobility_data.Mgb,
                        "Grain boundary diffusion mobility.");
      prm.add_parameter("L",
                        mobility_data.L,
                        "Grain boundary motion mobility.");
      prm.leave_subsection();


      prm.enter_subsection("TimeIntegration");
      prm.add_parameter("TimeStart",
                        time_integration_data.time_start,
                        "Start time.");
      prm.add_parameter("TimeEnd", time_integration_data.time_end, "End time.");
      prm.add_parameter("TimeStepInit",
                        time_integration_data.time_step_init,
                        "Initial timestep.");
      prm.add_parameter("TimeStepMin",
                        time_integration_data.time_step_min,
                        "Minimum timestep.");
      prm.add_parameter("TimeStepMax",
                        time_integration_data.time_step_max,
                        "Maximum timestep.");
      prm.add_parameter("GrowthFactor",
                        time_integration_data.growth_factor,
                        "Timestep growth factor.");
      prm.add_parameter("OutputTimeInterval",
                        time_integration_data.output_time_interval,
                        "Output time interval.");
      prm.add_parameter("DesirableNewtonIterations",
                        time_integration_data.desirable_newton_iterations,
                        "Desirable Newton iterations.");
      prm.add_parameter("DesirableLinearIterations",
                        time_integration_data.desirable_linear_iterations,
                        "Desirable linear iterations.");
      prm.leave_subsection();


      prm.enter_subsection("Preconditioners");
      const std::string preconditioner_types =
        "AMG|InverseBlockDiagonalMatrix|InverseDiagonalMatrix|ILU|InverseComponentBlockDiagonalMatrix";
      prm.add_parameter(
        "OuterPreconditioner",
        preconditioners_data.outer_preconditioner,
        "Preconditioner to be used for the outer system.",
        Patterns::Selection(
          preconditioner_types +
          "|BlockPreconditioner2|BlockPreconditioner3|BlockPreconditioner3CH"));

      prm.enter_subsection("BlockPreconditioner2");
      prm.add_parameter(
        "Block0Preconditioner",
        preconditioners_data.block_preconditioner_2_data.block_0_preconditioner,
        "Preconditioner to be used for the first block.",
        Patterns::Selection(preconditioner_types));
      prm.add_parameter(
        "Block1Preconditioner",
        preconditioners_data.block_preconditioner_2_data.block_1_preconditioner,
        "Preconditioner to be used for the second block.",
        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3");
      prm.add_parameter("Type",
                        preconditioners_data.block_preconditioner_3_data.type,
                        "Type of block preconditioner of CH system.",
                        Patterns::Selection("D|LD|RD|SYMM"));
      prm.add_parameter(
        "Block0Preconditioner",
        preconditioners_data.block_preconditioner_3_data.block_0_preconditioner,
        "Preconditioner to be used for the first block.",
        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block0RelativeTolerance",
                        preconditioners_data.block_preconditioner_3_data
                          .block_0_relative_tolerance,
                        "Relative tolerance of the first block.");
      prm.add_parameter(
        "Block1Preconditioner",
        preconditioners_data.block_preconditioner_3_data.block_1_preconditioner,
        "Preconditioner to be used for the second block.",
        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block1RelativeTolerance",
                        preconditioners_data.block_preconditioner_3_data
                          .block_1_relative_tolerance,
                        "Relative tolerance of the second block.");
      prm.add_parameter(
        "Block2Preconditioner",
        preconditioners_data.block_preconditioner_3_data.block_2_preconditioner,
        "Preconditioner to be used for the thrird block.",
        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2RelativeTolerance",
                        preconditioners_data.block_preconditioner_3_data
                          .block_2_relative_tolerance,
                        "Relative tolerance of the third block.");
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3CH");
      prm.add_parameter("Block0Preconditioner",
                        preconditioners_data.block_preconditioner_3_ch_data
                          .block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2Preconditioner",
                        preconditioners_data.block_preconditioner_3_ch_data
                          .block_2_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();

      prm.leave_subsection();
    }
  };
} // namespace Sintering