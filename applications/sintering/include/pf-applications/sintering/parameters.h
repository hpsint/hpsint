// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <deal.II/base/parameter_handler.h>

#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

#include <pf-applications/sintering/preconditioners.h>

#include <boost/algorithm/string.hpp>

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

  struct DivisionsData
  {
    unsigned int nx = 0;
    unsigned int ny = 0;
    unsigned int nz = 0;
  };

  struct GeometryData
  {
    double          divisions_per_interface            = 4;
    double          boundary_factor                    = 0.5;
    double          interface_width                    = 2.0;
    std::string     interface_direction                = "Middle";
    bool            minimize_order_parameters          = true;
    double          interface_buffer_ratio             = 1.0;
    double          radius_buffer_ratio                = 0.0;
    bool            periodic                           = false;
    bool            custom_bounding_box                = false;
    unsigned int    max_prime                          = 20;
    std::string     global_refinement                  = "None";
    std::string     initial_mesh                       = "Interface";
    double          max_level0_divisions_per_interface = 1.0 - 1e-9;
    BoundingBoxData bounding_box_data;
    DivisionsData   divisions_data;

    double hanging_node_weight = 1.0;
  };

  struct AdaptivityData
  {
    double       top_fraction_of_cells    = 0.3;
    double       bottom_fraction_of_cells = 0.1;
    unsigned int min_refinement_depth     = 3;
    unsigned int max_refinement_depth     = 0;
    unsigned int refinement_frequency     = 10; // 0 - no refinement
    bool         extra_coarsening         = false;
    double       interface_val_min        = 0.05;
    double       interface_val_max        = 0.95;

    bool   quality_control             = false;
    bool   auto_quality_min            = true;
    double auto_quality_initial_factor = 0.85;
    double quality_min                 = 0.5;
  };

  struct GrainTrackerData
  {
    double       threshold_lower         = 0.01;
    double       threshold_new_grains    = 0.02;
    double       buffer_distance_ratio   = 0.05;
    double       buffer_distance_fixed   = 0.0;
    unsigned int grain_tracker_frequency = 10; // 0 - no grain tracker
    unsigned int verbosity               = 0;

    bool fast_reassignment   = false;
    bool track_with_quality  = false;
    bool use_old_remap       = false;
    bool check_inconsistency = false;

    std::string grain_representation = "Spherical";
  };

  struct EnergyAbstractData
  {
    double A       = 16;
    double B       = 1;
    double kappa_c = 1;
    double kappa_p = 0.5;
  };

  struct EnergyRealisticData
  {
    double surface_energy        = 0;
    double grain_boundary_energy = 0;
  };

  struct MobilityAbstractData
  {
    double Mvol  = 1e-2;
    double Mvap  = 1e-10;
    double Msurf = 4;
    double Mgb   = 0.4;
    double L     = 1;
  };

  struct MobilityRealisticData
  {
    double omega = 0;

    double D_vol0  = 0;
    double D_vap0  = 0;
    double D_surf0 = 0;
    double D_gb0   = 0;

    double Q_vol  = 0;
    double Q_vap  = 0;
    double Q_surf = 0;
    double Q_gb   = 0;

    double D_gb_mob0 = 0;
    double Q_gb_mob  = 0;

    std::string arrhenius_unit = "Boltzmann";
  };

  struct MechanicsData
  {
    double E  = 1.0;
    double nu = 0.25;

    std::string plane_type = "None";

    double c_min = 0.1;
  };

  struct MaterialData
  {
    std::string type = "Abstract";

    double time_scale   = 1.;
    double length_scale = 1.;
    double energy_scale = 1.;

    std::map<double, double> temperature = {{0, 1573}, {100, 1573}};

    EnergyAbstractData  energy_abstract_data;
    EnergyRealisticData energy_realistic_data;

    MobilityAbstractData  mobility_abstract_data;
    MobilityRealisticData mobility_realistic_data;

    MechanicsData mechanics_data;
  };

  struct AdvectionData
  {
    bool enable = false;

    double k           = 100.;
    double mt          = 1.;
    double mr          = 1.;
    double cgb         = 0.;
    double ceq         = 1.;
    double smoothening = 0.;

    bool check_courant = true;
  };

  struct BoundaryConditionsData
  {
    std::string            type      = "Domain";
    std::set<unsigned int> direction = {0};

    bool prefer_growing = false;
    bool use_barycenter = false;
  };

  struct TimeIntegrationData
  {
    std::string integration_scheme = "BDF2";
    std::string predictor          = "None";

    double       time_start                  = 0;
    double       time_end                    = 1e3;
    double       time_step_init              = 1e-3;
    double       time_step_min               = 1e-5;
    double       time_step_max               = 1e2;
    double       growth_factor               = 1.2;
    unsigned int desirable_newton_iterations = 5;
    unsigned int desirable_linear_iterations = 100;
    bool         sanity_check_predictor      = false;
    bool         sanity_check_solution       = false;
  };

  struct OutputData
  {
    bool                  regular                = true;
    bool                  contours               = true;
    bool                  contours_tex           = false;
    unsigned int          n_coarsening_steps     = 0;
    unsigned int          n_mca_subdivisions     = 1;
    bool                  porosity               = false;
    bool                  porosity_contours      = false;
    bool                  porosity_smooth        = true;
    bool                  porosity_stats         = false;
    double                porosity_max_value     = 0.8;
    bool                  shrinkage              = false;
    unsigned int          shrinkage_intervals    = 20;
    bool                  grains_stats           = false;
    bool                  quality                = false;
    bool                  table                  = false;
    bool                  debug                  = false;
    bool                  higher_order_cells     = false;
    bool                  fluxes_divergences     = false;
    bool                  concentration_contour  = false;
    bool                  coordination_number    = false;
    bool                  grains_as_spheres      = false;
    bool                  particle_indices       = false;
    double                output_time_interval   = 10;
    std::string           vtk_path               = ".";
    std::set<std::string> fields                 = {"CH",
                                                    "AC",
                                                    "displ",
                                                    "bnds",
                                                    "gb",
                                                    "d2f",
                                                    "M",
                                                    "dM",
                                                    "kappa",
                                                    "L",
                                                    "flux",
                                                    "energy",
                                                    "subdomain",
                                                    "mf_indices",
                                                    "vel",
                                                    "trans"};
    bool                  mesh_overhead_estimate = false;
    bool                  only_control_boxes     = false;

    std::set<std::string> domain_integrals = {"gb_area",
                                              "solid_vol",
                                              "surf_area",
                                              "free_energy"};
    bool                  grain_boundaries = false;
    bool                  iso_surf_area    = false;
    bool                  iso_gb_area      = false;
    bool                  quality_min      = false;
    double                gb_threshold     = 0.14;
    bool                  auto_control_box = false;
    double                box_rel_padding  = 0.05;

    std::vector<std::pair<Point<3>, Point<3>>> control_boxes;

    std::vector<std::pair<std::string, double>> sections;
  };

  struct RestartData
  {
    std::string  prefix          = "./restart";
    std::string  type            = "never";
    double       interval        = 10.0;
    unsigned int max_output      = 0;
    bool         flexible_output = false;
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

  struct Parameters
  {
    ApproximationData      approximation_data;
    GeometryData           geometry_data;
    AdaptivityData         adaptivity_data;
    GrainTrackerData       grain_tracker_data;
    MaterialData           material_data;
    AdvectionData          advection_data;
    BoundaryConditionsData boundary_conditions;
    TimeIntegrationData    time_integration_data;
    OutputData             output_data;
    RestartData            restart_data;
    PreconditionersData    preconditioners_data;
    ProfilingData          profiling_data;

    NonLinearSolvers::NonLinearData nonlinear_data;

    bool   matrix_based                               = false;
    double grain_cut_off_tolerance                    = 0.0; // 0.00001
    bool   use_tensorial_mobility_gradient_on_the_fly = false;

    bool print_time_loop = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      prm.parse_input(file_name, "", true);
    }

    void
    set(const std::string param, const std::string val)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      std::vector<std::string> param_path;
      boost::split(param_path, param, boost::is_any_of("."));

      for (auto it = param_path.begin(); it != param_path.end(); ++it)
        if (std::distance(it, param_path.end()) > 1)
          prm.enter_subsection(*it);
        else
          prm.set(*it, val);
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
    print_help(std::ostream &out = std::cout)
    {
      print(out,
            ParameterHandler::OutputStyle::Description |
              ParameterHandler::OutputStyle::KeepDeclarationOrder);
    }

    void
    print_input(std::ostream &out = std::cout)
    {
      print(out, ParameterHandler::OutputStyle::ShortJSON);
    }

    void
    print(const ParameterHandler::OutputStyle style)
    {
      print(std::cout, style);
    }

    void
    print(std::ostream &out, const ParameterHandler::OutputStyle style)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      ConditionalOStream pcout(
        out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      if (pcout.is_active())
        prm.print_parameters(pcout.get_stream(), style);
    }

  private:
    void
    add_parameters(ParameterHandler &prm)
    {
      prm.add_parameter("MatrixBased",
                        matrix_based,
                        "Run program matrix-based or matrix-free.");
      prm.add_parameter("GrainCutOffTolerance",
                        grain_cut_off_tolerance,
                        "Grain cut-off tolerance.");
      prm.add_parameter("TensorialMobilityGradientOnTheFly",
                        use_tensorial_mobility_gradient_on_the_fly,
                        "Run program matrix-based or matrix-free.");
      prm.add_parameter("PrintTimeLoop",
                        print_time_loop,
                        "Generate output during time loop.");

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
      prm.add_parameter("DivisionsPerInterface",
                        geometry_data.divisions_per_interface,
                        "Number of divisions per interface.");
      prm.add_parameter("BoundaryFactor",
                        geometry_data.boundary_factor,
                        "Bounding box padding ratio (to the largest radius).");
      prm.add_parameter("InterfaceWidth",
                        geometry_data.interface_width,
                        "Phase-field interface width.");
      prm.add_parameter("InterfaceDirection",
                        geometry_data.interface_direction,
                        "Interface offset.",
                        Patterns::Selection("Inside|Outside|Middle"));
      prm.add_parameter("MinimizeOrderParameters",
                        geometry_data.minimize_order_parameters,
                        "Minimize number of initial order parameters.");
      prm.add_parameter("InterfaceBufferRatio",
                        geometry_data.interface_buffer_ratio,
                        "Interface buffer ratio.");
      prm.add_parameter("RadiusBufferRatio",
                        geometry_data.radius_buffer_ratio,
                        "Max radius based buffer ratio.");
      prm.add_parameter("Periodic",
                        geometry_data.periodic,
                        "Is domain periodic.");
      prm.add_parameter("CustomBoundingBox",
                        geometry_data.custom_bounding_box,
                        "Is custom bounding box specified.");
      prm.add_parameter("MaxPrime",
                        geometry_data.max_prime,
                        "Max prime number for subdivisions decomposition.");
      prm.add_parameter("GlobalRefinement",
                        geometry_data.global_refinement,
                        "Perform global refinements.",
                        Patterns::Selection("None|Base|Full"));
      prm.add_parameter("InitialMesh",
                        geometry_data.initial_mesh,
                        "Method for building initial mesh.",
                        Patterns::Selection("Interface|MaxRadius|Divisions"));
      prm.add_parameter("MaxLevel0DivisionsPerInterface",
                        geometry_data.max_level0_divisions_per_interface,
                        "Maximum initial number of divisions per interface.");

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

      prm.enter_subsection("Divisions");
      prm.add_parameter("Nx", geometry_data.divisions_data.nx, "nx.");
      prm.add_parameter("Ny", geometry_data.divisions_data.ny, "ny.");
      prm.add_parameter("Nz", geometry_data.divisions_data.nz, "nz.");
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
      prm.add_parameter("QualityControl",
                        adaptivity_data.quality_control,
                        "Control automatically mesh quality.");
      prm.add_parameter("AutoQualityMin",
                        adaptivity_data.auto_quality_min,
                        "Automatically set up minimun quality.");
      prm.add_parameter("AutoQualityInitialFactor",
                        adaptivity_data.auto_quality_initial_factor,
                        "Reduction factor to set up quality threshold.");
      prm.add_parameter("QualityMin",
                        adaptivity_data.quality_min,
                        "Minimum value for cell quality (0 - low, 1 - high).");
      prm.add_parameter("ExtraCoarsening",
                        adaptivity_data.extra_coarsening,
                        "Allow reduce quality of the mesh.");
      prm.add_parameter("InterfaceValueMin",
                        adaptivity_data.interface_val_min,
                        "Minimum value at the interface.");
      prm.add_parameter("InterfaceValueMax",
                        adaptivity_data.interface_val_max,
                        "Maximum value at the interface.");
      prm.leave_subsection();


      prm.enter_subsection("GrainTracker");
      prm.add_parameter("ThresholdLower",
                        grain_tracker_data.threshold_lower,
                        "Lower boundary for detecting grain.");
      prm.add_parameter("ThresholdNewGrains",
                        grain_tracker_data.threshold_new_grains,
                        "Lower boundary for detecting new grain.");
      prm.add_parameter("BufferDistanceRatio",
                        grain_tracker_data.buffer_distance_ratio,
                        "Ratio of the transfer buffer (to the grain radius).");
      prm.add_parameter("BufferDistanceFixed",
                        grain_tracker_data.buffer_distance_fixed,
                        "Fixed size of the transfer buffer.");
      prm.add_parameter("GrainTrackerFrequency",
                        grain_tracker_data.grain_tracker_frequency,
                        "Grain tracker frequency (0 = no grain tracking).");
      prm.add_parameter("Verbosity",
                        grain_tracker_data.verbosity,
                        "Grain tracker verbosity level.");
      prm.add_parameter("FastReassignment",
                        grain_tracker_data.fast_reassignment,
                        "Use fast grain reassignment strategy.");
      prm.add_parameter(
        "TrackWithQuality",
        grain_tracker_data.track_with_quality,
        "Run grain tracker if the mesh refinement is triggered by the quality control.");
      prm.add_parameter("UseOldRemap",
                        grain_tracker_data.use_old_remap,
                        "Use old remapping algo.");
      prm.add_parameter("CheckInconsistency",
                        grain_tracker_data.check_inconsistency,
                        "Check grains inconsistency at the end of timestep.");
      prm.add_parameter("GrainRepresentation",
                        grain_tracker_data.grain_representation,
                        "Grain representation",
                        Patterns::Selection("Spherical|Elliptical|Wavefront"));
      prm.leave_subsection();


      prm.enter_subsection("Material");
      prm.add_parameter("Type",
                        material_data.type,
                        "Material type.",
                        Patterns::Selection("Abstract|Realistic"));
      prm.add_parameter("TimeScale", material_data.time_scale, "Time scale.");
      prm.add_parameter("LengthScale",
                        material_data.length_scale,
                        "Length scale.");
      prm.add_parameter("EnergyScale",
                        material_data.energy_scale,
                        "Energy scale.");
      prm.add_parameter("Temperature",
                        material_data.temperature,
                        "Temperature profile.");

      prm.enter_subsection("EnergyAbstract");
      prm.add_parameter("A",
                        material_data.energy_abstract_data.A,
                        "Energy parameter A.");
      prm.add_parameter("B",
                        material_data.energy_abstract_data.B,
                        "Energy parameter B.");
      prm.add_parameter("KappaC",
                        material_data.energy_abstract_data.kappa_c,
                        "Barrier height kappa_c.");
      prm.add_parameter("KappaP",
                        material_data.energy_abstract_data.kappa_p,
                        "Barrier height kappa_p.");
      prm.leave_subsection();

      prm.enter_subsection("MobilityAbstract");
      prm.add_parameter("Mvol",
                        material_data.mobility_abstract_data.Mvol,
                        "Volumetric diffusion mobility.");
      prm.add_parameter("Mvap",
                        material_data.mobility_abstract_data.Mvap,
                        "Vaporization diffusion mobility.");
      prm.add_parameter("Msurf",
                        material_data.mobility_abstract_data.Msurf,
                        "Surface diffusion mobility.");
      prm.add_parameter("Mgb",
                        material_data.mobility_abstract_data.Mgb,
                        "Grain boundary diffusion mobility.");
      prm.add_parameter("L",
                        material_data.mobility_abstract_data.L,
                        "Grain boundary motion mobility.");
      prm.leave_subsection();

      prm.enter_subsection("EnergyRealistic");
      prm.add_parameter("SurfaceEnergy",
                        material_data.energy_realistic_data.surface_energy,
                        "Surface energy.");
      prm.add_parameter(
        "GrainBoundaryEnergy",
        material_data.energy_realistic_data.grain_boundary_energy,
        "Grain boundary energy.");
      prm.leave_subsection();

      prm.enter_subsection("MobilityRealistic");
      prm.add_parameter("Omega",
                        material_data.mobility_realistic_data.omega,
                        "Atomic volume.");

      prm.add_parameter("DVol0",
                        material_data.mobility_realistic_data.D_vol0,
                        "Volumetric diffusion mobility prefactor.");
      prm.add_parameter("DVap0",
                        material_data.mobility_realistic_data.D_vap0,
                        "Vaporization diffusion mobility prefactor.");
      prm.add_parameter("DSurf0",
                        material_data.mobility_realistic_data.D_surf0,
                        "Surface diffusion mobility prefactor.");
      prm.add_parameter("DGb0",
                        material_data.mobility_realistic_data.D_gb0,
                        "Grain boundary diffusion mobility prefactor.");

      prm.add_parameter("QVol",
                        material_data.mobility_realistic_data.Q_vol,
                        "Volumetric diffusion activation energy.");
      prm.add_parameter("QVap",
                        material_data.mobility_realistic_data.Q_vap,
                        "Vaporization diffusion activation energy.");
      prm.add_parameter("QSurf",
                        material_data.mobility_realistic_data.Q_surf,
                        "Surface diffusion activation energy.");
      prm.add_parameter("QGb",
                        material_data.mobility_realistic_data.Q_gb,
                        "Grain boundary diffusion activation energy.");

      prm.add_parameter("DGbMob0",
                        material_data.mobility_realistic_data.D_gb_mob0,
                        "Grain boundary mobility prefactor.");
      prm.add_parameter("QGbMob",
                        material_data.mobility_realistic_data.Q_gb_mob,
                        "Grain boundary mobility activation energy.");

      prm.leave_subsection();

      prm.enter_subsection("Mechanics");
      prm.add_parameter("E", material_data.mechanics_data.E, "Young modulus.");
      prm.add_parameter("nu",
                        material_data.mechanics_data.nu,
                        "Poisson ratio.");
      prm.add_parameter("Type",
                        material_data.mechanics_data.plane_type,
                        "Type of material for 2D case",
                        Patterns::Selection("None|PlaneStrain|PlaneStress"));
      prm.add_parameter("Cmin",
                        material_data.mechanics_data.c_min,
                        "Minimum value of concentration.");
      prm.leave_subsection();

      prm.leave_subsection();


      prm.enter_subsection("Advection");
      prm.add_parameter("Enable",
                        advection_data.enable,
                        "Enable Wang mechanism");
      prm.add_parameter("K", advection_data.k, "Wang stiffness.");
      prm.add_parameter("Mt", advection_data.mt, "Wang translation factor.");
      prm.add_parameter("Mr", advection_data.mr, "Wang rotation factor.");
      prm.add_parameter("Cgb", advection_data.cgb, "Grain boundary threshold.");
      prm.add_parameter("Ceq",
                        advection_data.ceq,
                        "Grain boundary equilibrium concentration.");
      prm.add_parameter(
        "Smoothening",
        advection_data.smoothening,
        "Heaviside smoothening, recommended range 50...5000, 0 = disabled (default).");
      prm.add_parameter("CheckCourant",
                        advection_data.check_courant,
                        "Check Courant condition at the end of the timestep");
      prm.leave_subsection();


      prm.enter_subsection("BoundaryConditions");
      prm.add_parameter(
        "Type",
        boundary_conditions.type,
        "Type of boundary conditions for the coupled model",
        Patterns::Selection(
          "None|CentralSection|CentralParticle|CentralParticleSection|Domain"));
      const std::string bc_direction_options = "0|1|2";
      prm.add_parameter("Direction",
                        boundary_conditions.direction,
                        "Primary direction for restraining displacements.",
                        Patterns::List(
                          Patterns::MultipleSelection(bc_direction_options)));
      prm.add_parameter("PreferGrowing",
                        boundary_conditions.prefer_growing,
                        "Prefer growing grains for BC imposition.");
      prm.add_parameter("UseBarycenter",
                        boundary_conditions.use_barycenter,
                        "Use cloud barycenter for the reference grain search.");
      prm.leave_subsection();


      prm.enter_subsection("TimeIntegration");
      prm.add_parameter("IntegrationScheme",
                        time_integration_data.integration_scheme,
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
      prm.add_parameter("ContourTex",
                        output_data.contours_tex,
                        "Whether contour output is enabled.");
      prm.add_parameter("ContourNCoarseningSteps",
                        output_data.n_coarsening_steps,
                        "Whether contour output is enabled.");
      prm.add_parameter("MCASubdivisions",
                        output_data.n_mca_subdivisions,
                        "Marching cube algorithm subdivisions.");
      prm.add_parameter("Porosity",
                        output_data.porosity,
                        "Output porosity to VTK.");
      prm.add_parameter("PorosityContours",
                        output_data.porosity_contours,
                        "Output porosity contours to VTK.");
      prm.add_parameter("PorositySmooth",
                        output_data.porosity_smooth,
                        "Whether to output smooth porosity to VTK.");
      prm.add_parameter("PorosityStats",
                        output_data.porosity_stats,
                        "Determine porosity and output its stats.");
      prm.add_parameter("PorosityMaxValue",
                        output_data.porosity_max_value,
                        "Maximum value of concentration inside pores.");
      prm.add_parameter("Shrinkage",
                        output_data.shrinkage,
                        "Determine shrinkage.");
      prm.add_parameter("ShrinkageIntervals",
                        output_data.shrinkage_intervals,
                        "Discretization intervals for shrinkage accuracy.");
      prm.add_parameter("GrainsStats",
                        output_data.grains_stats,
                        "Grains statistics.");
      prm.add_parameter("Quality", output_data.quality, "Output mesh quality.");
      prm.add_parameter("Table", output_data.table, "Output table.");
      prm.add_parameter("Debug",
                        output_data.debug,
                        "Whether debug output is enabled.");
      prm.add_parameter("HigherOrderCells",
                        output_data.higher_order_cells,
                        "Use higher order cells.");
      prm.add_parameter("FluxesDivergences",
                        output_data.fluxes_divergences,
                        "Calculate divergences of fluxes.");
      prm.add_parameter("ConcentrationContour",
                        output_data.concentration_contour,
                        "Plot concentration contour.");
      prm.add_parameter("CoordinationNumber",
                        output_data.coordination_number,
                        "Compute coordination number.");
      prm.add_parameter("GrainsAsSpheres",
                        output_data.grains_as_spheres,
                        "Output grains as spheres using particles framework.");
      prm.add_parameter("ParticleIndices",
                        output_data.particle_indices,
                        "Internal particle indices of the grain tracker.");
      prm.add_parameter("OutputTimeInterval",
                        output_data.output_time_interval,
                        "Output time interval.");
      prm.add_parameter("VtkPath",
                        output_data.vtk_path,
                        "Path to write VTK files.");
      const std::string output_fields_options =
        "CH|AC|displ|bnds|gb|d2f|M|dM|kappa|L|energy|flux|subdomain|mf_indices|vel|trans|strain_lin";
      prm.add_parameter("Fields",
                        output_data.fields,
                        "Fields to output.",
                        Patterns::List(
                          Patterns::MultipleSelection(output_fields_options)));
      prm.add_parameter("MeshOverheadEstimate",
                        output_data.mesh_overhead_estimate,
                        "Print mesh overhead estimate.");
      prm.add_parameter("OnlyControlBoxes",
                        output_data.only_control_boxes,
                        "Output only for control boxes.");
      const std::string domain_integrals_options =
        "gb_area|solid_vol|surf_area|avg_grain_size|surf_area_nrm|free_energy|bulk_energy|interface_energy|order_params|control_vol";
      prm.add_parameter("DomainIntegrals",
                        output_data.domain_integrals,
                        "Domain integral quantities.",
                        Patterns::List(Patterns::MultipleSelection(
                          domain_integrals_options)));

      std::vector<std::unique_ptr<Patterns::PatternBase>> ps;
      ps.template emplace_back(
        Patterns::Tools::Convert<Point<3>>::to_pattern());
      ps.template emplace_back(
        Patterns::Tools::Convert<Point<3>>::to_pattern());

      prm.add_parameter("ControlBoxes",
                        output_data.control_boxes,
                        "Control boxes to use for output.",
                        Patterns::List(Patterns::Tuple(ps, ";"),
                                       0,
                                       Patterns::List::max_int_value,
                                       "|"));

      std::vector<std::unique_ptr<Patterns::PatternBase>> psec;
      psec.push_back(std::make_unique<Patterns::Anything>());
      psec.push_back(std::make_unique<Patterns::Double>());

      prm.add_parameter("Sections",
                        output_data.sections,
                        "Cross sections for projection output.",
                        Patterns::List(Patterns::Tuple(psec, "="),
                                       0,
                                       Patterns::List::max_int_value,
                                       "|"));

      prm.add_parameter("GrainBoundaries",
                        output_data.grain_boundaries,
                        "Output grain boundaries.");
      prm.add_parameter("IsoSurfaceArea",
                        output_data.iso_surf_area,
                        "Compute surface area from isocontours.");
      prm.add_parameter("IsoGrainBoundariesArea",
                        output_data.iso_gb_area,
                        "Compute GB area from isocontours.");
      prm.add_parameter("QualityMin",
                        output_data.quality_min,
                        "Output min mesh quality.");
      prm.add_parameter("GrainBoundariesThreshold",
                        output_data.gb_threshold,
                        "Grain boundary detection threshold.");
      prm.add_parameter("AutoControlBox",
                        output_data.auto_control_box,
                        "Add automatic control box.");
      prm.add_parameter(
        "BoxRelativePadding",
        output_data.box_rel_padding,
        "Auto control box padding relative to the packing size.");

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
      prm.add_parameter("LinearSolver",
                        nonlinear_data.l_solver,
                        "Name of linear solver.",
                        Patterns::Selection(
                          "GMRES|IDR|Bicgstab|Relaxation|Direct"));
      prm.add_parameter("LinearSolverBicgstabTries",
                        nonlinear_data.l_bisgstab_tries,
                        "Number of Bicgstab before switching to GMRES.");

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
                        Patterns::Selection("damped|NOX|SNES"));

      prm.add_parameter("FDMJacobianApproximation",
                        nonlinear_data.fdm_jacobian_approximation);
      prm.add_parameter("FDMPrecondSystemApproximation",
                        nonlinear_data.fdm_precond_system_approximation);
      prm.add_parameter("JacobiFree", nonlinear_data.jacobi_free);
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

      prm.enter_subsection("SNESData");

      // see: https://petsc.org/main/docs/manual/snes/#the-nonlinear-solvers
      prm.add_parameter("SolverName",
                        nonlinear_data.snes_data.solver_name,
                        "SNES solver name",
                        Patterns::Selection(
                          "newtonls|newtontr|ngmres|anderson"));

      // see: https://petsc.org/main/docs/manual/snes/#line-search-newton
      prm.add_parameter("LineSearchName",
                        nonlinear_data.snes_data.line_search_name,
                        "SNES line search algorithm name",
                        Patterns::Selection("bt|basic|none|l2|cp"));
      prm.leave_subsection();

      prm.enter_subsection("GMRESData");
      prm.add_parameter("OrthogonalizationStrategy",
                        nonlinear_data.gmres_data.orthogonalization_strategy,
                        "Orthogonalization strategy",
                        Patterns::Selection(
                          "classical gram schmidt|modified gram schmidt"));
      prm.leave_subsection();

      prm.leave_subsection();

      prm.enter_subsection("Preconditioners");
      const std::string preconditioner_types =
        "AMG|BlockAMG|BlockILU|InverseBlockDiagonalMatrix|InverseDiagonalMatrix|ILU|InverseComponentBlockDiagonalMatrix|BlockGMG|GMG|Identity";
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
      prm.add_parameter(
        "Block2Preconditioner",
        preconditioners_data.block_preconditioner_2_data.block_2_preconditioner,
        "Preconditioner to be used for the first block.",
        Patterns::Selection(preconditioner_types));

      prm.enter_subsection("Block2AMG");
      prm.add_parameter("SmootherSweeps",
                        preconditioners_data.block_preconditioner_2_data
                          .block_2_amg_data.smoother_sweeps,
                        "Smoother sweeps");
      prm.add_parameter("NCycles",
                        preconditioners_data.block_preconditioner_2_data
                          .block_2_amg_data.n_cycles,
                        "Number of cycles");
      prm.leave_subsection();

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