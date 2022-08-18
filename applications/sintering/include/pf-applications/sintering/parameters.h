#pragma once

#include <deal.II/base/parameter_handler.h>

#include <pf-applications/sintering/preconditioners.h>

namespace Sintering
{
  struct ApproximationData
  {
    unsigned int fe_degree      = 1;
    unsigned int n_subdivisions = 1;
    unsigned int n_points_1D    = 2;
  };

  struct BoundingBoxData
  {
    double x_min = 0;
    double x_max = 0;
    double y_min = 0;
    double y_max = 0;
    double z_min = 0;
    double z_max = 0;
  };

  struct GeometryData
  {
    unsigned int    elements_per_interface = 8; // 4 - works well with AMR=off
    double          boundary_factor        = 0.5;
    double          interface_width        = 2.0;
    bool            minimize_order_parameters = true;
    double          interface_buffer_ratio    = 1.0;
    bool            periodic                  = false;
    bool            custom_bounding_box       = false;
    BoundingBoxData bounding_box_data;

    double hanging_node_weight = 1.0;
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
    std::string interation_scheme = "BDF1";
    std::string predictor         = "Euler";

    double       time_start                  = 0;
    double       time_end                    = 1e3;
    double       time_step_init              = 1e-3;
    double       time_step_min               = 1e-5;
    double       time_step_max               = 1e2;
    double       growth_factor               = 1.2;
    unsigned int desirable_newton_iterations = 5;
    unsigned int desirable_linear_iterations = 100;
    bool         sanity_check_predictor      = true;
    bool         sanity_check_solution       = true;
  };

  struct OutputData
  {
    bool                  regular              = true;
    bool                  contours             = true;
    unsigned int          n_coarsening_steps   = 0;
    bool                  porosity             = false;
    bool                  shrinkage            = false;
    bool                  debug                = false;
    bool                  higher_order_cells   = false;
    double                output_time_interval = 10;
    std::string           vtk_path             = ".";
    std::set<std::string> fields =
      {"CH", "AC", "bnds", "dt", "d2f", "M", "dM", "kappa", "L", "subdomain"};
    bool mesh_overhead_estimate = false;
  };

  struct RestartData
  {
    std::string  prefix          = "./restart";
    std::string  type            = "never";
    double       interval        = 10.0;
    unsigned int max_output      = 0;
    bool         flexible_output = true;
    bool         full_history    = true;
  };

  struct PreconditionersData
  {
    std::string outer_preconditioner = "BlockPreconditioner2";
    // std::string outer_preconditioner = "ILU";

    BlockPreconditioner2Data block_preconditioner_2_data;
  };

  struct ProfilingData
  {
    bool   run_vmults                = false;
    double output_time_interval      = -1.0; // default: never
    bool   output_memory_consumption = false;
  };

  struct NOXData
  {
    int         output_information             = 0;
    std::string direction_method               = "Newton";
    std::string line_search_method             = "Full Step";
    std::string line_search_interpolation_type = "Cubic";
  };

  struct NonLinearData
  {
    int    nl_max_iter = 10;
    double nl_abs_tol  = 1.e-20;
    double nl_rel_tol  = 1.e-5;

    int    l_max_iter = 1000;
    double l_abs_tol  = 1.e-10;
    double l_rel_tol  = 1.e-2;

    bool         newton_do_update             = true;
    unsigned int newton_threshold_newton_iter = 100;
    unsigned int newton_threshold_linear_iter = 20;
    bool         newton_reuse_preconditioner  = true;
    bool         newton_use_damping           = true;

    std::string nonlinear_solver_type = "damped";

    unsigned int verbosity = 1;

    NOXData nox_data;
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
    OutputData          output_data;
    RestartData         restart_data;
    PreconditionersData preconditioners_data;
    ProfilingData       profiling_data;
    NonLinearData       nonlinear_data;

    bool matrix_based = false;

    bool print_time_loop = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      prm.parse_input(file_name, "", true);
    }

    void
    check()
    {
#ifdef FE_DEGREE
      if (approximation_data.n_subdivisions == 1)
        {
          AssertThrow(FE_DEGREE == approximation_data.fe_degree,
                      StandardExceptions::ExcDimensionMismatch(
                        FE_DEGREE, approximation_data.fe_degree));
        }
      else
        {
          AssertThrow(FE_DEGREE == approximation_data.n_subdivisions,
                      StandardExceptions::ExcDimensionMismatch(
                        FE_DEGREE, approximation_data.n_subdivisions + 1));
        }
#endif

#ifdef N_Q_POINTS_1D
      if (approximation_data.n_subdivisions == 1)
        {
          AssertThrow(N_Q_POINTS_1D == approximation_data.n_points_1D,
                      StandardExceptions::ExcDimensionMismatch(
                        N_Q_POINTS_1D, approximation_data.n_points_1D));
        }
      else
        {
          AssertThrow(N_Q_POINTS_1D == approximation_data.n_points_1D *
                                         approximation_data.n_subdivisions,
                      StandardExceptions::ExcDimensionMismatch(
                        N_Q_POINTS_1D,
                        approximation_data.n_points_1D *
                          approximation_data.n_subdivisions));
        }
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
      prm.add_parameter("MatrixBased",
                        matrix_based,
                        "Run program matrix-based or matrix-free.");

      prm.enter_subsection("Approximation");
      prm.add_parameter("FEDegree",
                        approximation_data.fe_degree,
                        "Degree of the shape the finite element.");
      prm.add_parameter("NSubdivisions",
                        approximation_data.n_subdivisions,
                        "Number of subdivisions.");
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
      prm.add_parameter("InterfaceBufferRatio",
                        geometry_data.interface_buffer_ratio,
                        "Interface buffer ratio.");
      prm.add_parameter("Periodic",
                        geometry_data.periodic,
                        "Is domain periodic.");
      prm.add_parameter("CustomBoundingBox",
                        geometry_data.custom_bounding_box,
                        "Is custom bounding box specified.");

      prm.enter_subsection("BoundingBox");
      prm.add_parameter("Xmin",
                        geometry_data.bounding_box_data.x_min,
                        "x min.");
      prm.add_parameter("Xmax",
                        geometry_data.bounding_box_data.x_max,
                        "x max.");
      prm.add_parameter("Ymin",
                        geometry_data.bounding_box_data.y_min,
                        "y min.");
      prm.add_parameter("Ymax",
                        geometry_data.bounding_box_data.y_max,
                        "y max.");
      prm.add_parameter("Zmin",
                        geometry_data.bounding_box_data.z_min,
                        "z min.");
      prm.add_parameter("Zmax",
                        geometry_data.bounding_box_data.z_max,
                        "z max.");
      prm.leave_subsection();

      prm.add_parameter("HangingNodeWeight",
                        geometry_data.hanging_node_weight,
                        "Factor to weight cells with hanging nodes with.");

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
      prm.add_parameter("IntegrationScheme",
                        time_integration_data.interation_scheme,
                        "Integration scheme.",
                        Patterns::Selection("BDF1|BDF2|BDF3"));
      prm.add_parameter("Predictor",
                        time_integration_data.predictor,
                        "Predictor for initial guess extrapolation.",
                        Patterns::Selection("None|Linear|Euler|Midpoint"));
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
      prm.add_parameter("DesirableNewtonIterations",
                        time_integration_data.desirable_newton_iterations,
                        "Desirable Newton iterations.");
      prm.add_parameter("DesirableLinearIterations",
                        time_integration_data.desirable_linear_iterations,
                        "Desirable linear iterations.");
      prm.add_parameter("SanityCheckPredictor",
                        time_integration_data.sanity_check_predictor,
                        "Whether to perform PF sanity check after predictor.");
      prm.add_parameter("SanityCheckSolution",
                        time_integration_data.sanity_check_solution,
                        "Whether to perform PF sanity check after solution.");
      prm.leave_subsection();

      prm.enter_subsection("Output");
      prm.add_parameter("Regular",
                        output_data.regular,
                        "Whether regular output is enabled.");
      prm.add_parameter("Contour",
                        output_data.contours,
                        "Whether contour output is enabled.");
      prm.add_parameter("ContourNCoarseningSteps",
                        output_data.n_coarsening_steps,
                        "Whether contour output is enabled.");
      prm.add_parameter("Porosity",
                        output_data.porosity,
                        "Determine porosity.");
      prm.add_parameter("Shrinkage",
                        output_data.shrinkage,
                        "Determine shrinkage.");
      prm.add_parameter("Debug",
                        output_data.debug,
                        "Whether debug output is enabled.");
      prm.add_parameter("HigherOrderCells",
                        output_data.higher_order_cells,
                        "Use higher order cells.");
      prm.add_parameter("OutputTimeInterval",
                        output_data.output_time_interval,
                        "Output time interval.");
      prm.add_parameter("VtkPath",
                        output_data.vtk_path,
                        "Path to write VTK files.");
      const std::string output_fields_options =
        "CH|AC|bnds|dt|d2f|M|dM|kappa|L|subdomain";
      prm.add_parameter("Fields",
                        output_data.fields,
                        "Fields to output.",
                        Patterns::List(
                          Patterns::MultipleSelection(output_fields_options)));
      prm.add_parameter("MeshOverheadEstimate",
                        output_data.mesh_overhead_estimate,
                        "Print mesh overhead estimate.");
      prm.leave_subsection();

      prm.enter_subsection("Restart");
      prm.add_parameter("Prefix",
                        restart_data.prefix,
                        "Prefix of restart files to create.");
      prm.add_parameter("Type",
                        restart_data.type,
                        "Type of restart output.",
                        Patterns::Selection(
                          "never|n_calls|real_time|simulation_time"));
      prm.add_parameter("Interval",
                        restart_data.interval,
                        "Interval of restart output.");
      prm.add_parameter(
        "FlexibleOutput",
        restart_data.flexible_output,
        "Allow flexible output. If enabled, you can restart with any number of "
        "processes but the generated file is (significanlty) larger.");
      prm.add_parameter(
        "FullHistory",
        restart_data.full_history,
        "Save full history. If enabled, all previous solutions are saved and "
        "then the higher order integration scheme can be applied.");
      prm.add_parameter(
        "MaximalOutput",
        restart_data.max_output,
        "Maximal number of restart outputs. The value 0 means no limit.");
      prm.leave_subsection();

      prm.enter_subsection("NonLinearData");
      prm.add_parameter("NonLinearMaxIterations", nonlinear_data.nl_max_iter);
      prm.add_parameter("NonLinearAbsoluteTolerance",
                        nonlinear_data.nl_abs_tol);
      prm.add_parameter("NonLinearRelativeTolerance",
                        nonlinear_data.nl_rel_tol);
      prm.add_parameter("LinearMaxIterations", nonlinear_data.l_max_iter);
      prm.add_parameter("LinearAbsoluteTolerance", nonlinear_data.l_abs_tol);
      prm.add_parameter("LinearRelativeTolerance", nonlinear_data.l_rel_tol);
      prm.add_parameter("NewtonDoUpdate", nonlinear_data.newton_do_update);
      prm.add_parameter("NewtonThresholdNewtonIterations",
                        nonlinear_data.newton_threshold_newton_iter);
      prm.add_parameter("NewtonThresholdLinearIterations",
                        nonlinear_data.newton_threshold_linear_iter);
      prm.add_parameter("NewtonReusePreconditioner",
                        nonlinear_data.newton_reuse_preconditioner);
      prm.add_parameter("NewtonUseDamping", nonlinear_data.newton_use_damping);

      prm.add_parameter("NonLinearSolverType",
                        nonlinear_data.nonlinear_solver_type,
                        "Type of the non-linear solver.",
                        Patterns::Selection("damped|NOX"));

      prm.add_parameter("Verbosity", nonlinear_data.verbosity);

      prm.enter_subsection("NOXData");
      prm.add_parameter("OutputInformation",
                        nonlinear_data.nox_data.output_information,
                        "NOX verbosity level");
      prm.add_parameter(
        "DirectionMethod",
        nonlinear_data.nox_data.direction_method,
        "How to compute the primary direction associated with the method",
        Patterns::Selection("Newton|Steepest Descent|NonlinearCG|Broyden"));
      prm.add_parameter("LineSearchMethod",
                        nonlinear_data.nox_data.line_search_method,
                        "Line search method",
                        Patterns::Selection("Full Step|Backtrack|Polynomial"));
      prm.add_parameter("LineSearchInterpolationType",
                        nonlinear_data.nox_data.line_search_interpolation_type,
                        "Polynomial line search interpolation type",
                        Patterns::Selection("Quadratic|Quadratic3|Cubic"));
      prm.leave_subsection();

      prm.leave_subsection();

      prm.enter_subsection("Preconditioners");
      const std::string preconditioner_types =
        "AMG|BlockAMG|BlockILU|InverseBlockDiagonalMatrix|InverseDiagonalMatrix|ILU|InverseComponentBlockDiagonalMatrix|BlockGMG|GMG";
      prm.add_parameter("OuterPreconditioner",
                        preconditioners_data.outer_preconditioner,
                        "Preconditioner to be used for the outer system.",
                        Patterns::Selection(preconditioner_types +
                                            "|BlockPreconditioner2"));

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
      prm.add_parameter(
        "Block1Approximation",
        preconditioners_data.block_preconditioner_2_data.block_1_approximation,
        "Approximation of the second block (Allen Cahn).",
        Patterns::Selection("all|const|max|avg"));
      prm.leave_subsection();

      prm.leave_subsection();

      prm.enter_subsection("Profiling");
      prm.add_parameter("RunVmults",
                        profiling_data.run_vmults,
                        "Run vmults standalone.");
      prm.add_parameter("OutputMemoryConsumption",
                        profiling_data.output_memory_consumption,
                        "Output memory consumption.");
      prm.add_parameter("OutputTimeInterval",
                        profiling_data.output_time_interval,
                        "Specify the inverval to print timings in seconds.");
      prm.leave_subsection();
    }
  };
} // namespace Sintering