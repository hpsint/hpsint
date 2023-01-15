// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/base/fe_integrator.h>
#include <pf-applications/base/solution_serialization.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/data_out.h>
#include <pf-applications/numerics/vector_tools.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/boundary_conditions.h>
#include <pf-applications/sintering/creator.h>
#include <pf-applications/sintering/initial_values.h>
#include <pf-applications/sintering/operator_advection.h>
#include <pf-applications/sintering/operator_postproc.h>
#include <pf-applications/sintering/operator_sintering_coupled_diffusion.h>
#include <pf-applications/sintering/operator_sintering_coupled_wang.h>
#include <pf-applications/sintering/operator_sintering_generic.h>
#include <pf-applications/sintering/parameters.h>
#include <pf-applications/sintering/postprocessors.h>
#include <pf-applications/sintering/preconditioners.h>
#include <pf-applications/sintering/tools.h>

#include <deal.II/trilinos/nox.h>
#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/grid/constraint_helper.h>

// Available sintering operators
#define OPERATOR_GENERIC 1
#define OPERATOR_COUPLED_WANG 2
#define OPERATOR_COUPLED_DIFFUSION 3

namespace Sintering
{
  using namespace dealii;


  template <int dim,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class Problem
  {
  public:
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    // Choose sintering operator
    using NonLinearOperator =
#if OPERATOR == OPERATOR_GENERIC
      SinteringOperatorGeneric<dim, Number, VectorizedArrayType>;
#elif OPERATOR == OPERATOR_COUPLED_WANG
      SinteringOperatorCoupledWang<dim, Number, VectorizedArrayType>;
#elif OPERATOR == OPERATOR_COUPLED_DIFFUSION
      SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>;
#else
#  error "Option OPERATOR has to be specified"
#endif

    const Parameters                          params;
    ConditionalOStream                        pcout;
    ConditionalOStream                        pcout_statistics;
    parallel::distributed::Triangulation<dim> tria;
    std::shared_ptr<const FiniteElement<dim>> fe;
    MappingQ<dim>                             mapping;
    Quadrature<1>                             quad;
    DoFHandler<dim>                           dof_handler;

    std::unique_ptr<dealii::parallel::Helper<dim>> helper;

    AffineConstraints<Number> constraints;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    // multigrid
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;

    MGLevelObject<DoFHandler<dim>>           mg_dof_handlers;
    MGLevelObject<AffineConstraints<Number>> mg_constraints;
    MGLevelObject<MGTwoLevelTransfer<dim, typename VectorType::BlockType>>
                                                                transfers;
    MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>> mg_matrix_free;

    std::shared_ptr<
      MGTransferGlobalCoarsening<dim, typename VectorType::BlockType>>
      transfer;


    std::pair<dealii::Point<dim>, dealii::Point<dim>>
           geometry_domain_boundaries;
    double geometry_r_max;
    double geometry_interface_width;

    unsigned int n_global_levels_0;
    double       time_last_output;
    unsigned int n_timestep;
    unsigned int n_linear_iterations;
    unsigned int n_non_linear_iterations;
    unsigned int n_residual_evaluations;
    unsigned int n_failed_tries;
    unsigned int n_failed_linear_iterations;
    unsigned int n_failed_non_linear_iterations;
    unsigned int n_failed_residual_evaluations;
    double       max_reached_dt;
    unsigned int restart_counter;
    double       t;

    std::vector<Number> dts{0};

    std::map<std::string, unsigned int> counters;

    Problem(const Parameters &                  params,
            std::shared_ptr<InitialValues<dim>> initial_solution)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , mapping(1)
      , quad(QIterated<1>(QGauss<1>(params.approximation_data.n_points_1D),
                          params.approximation_data.n_subdivisions))
      , dof_handler(tria)
    {
      MyScope("Problem::constructor");

      fe = create_fe(params.approximation_data.fe_degree,
                     params.approximation_data.n_subdivisions);

      geometry_domain_boundaries    = initial_solution->get_domain_boundaries();
      geometry_r_max                = initial_solution->get_r_max();
      geometry_interface_width      = initial_solution->get_interface_width();
      this->time_last_output        = 0;
      this->n_timestep              = 0;
      this->n_linear_iterations     = 0;
      this->n_non_linear_iterations = 0;
      this->n_residual_evaluations  = 0;
      this->n_failed_tries          = 0;
      this->n_failed_linear_iterations     = 0;
      this->n_failed_non_linear_iterations = 0;
      this->n_failed_residual_evaluations  = 0;
      this->max_reached_dt                 = 0.0;
      this->restart_counter                = 0;
      this->t                              = 0;
      this->counters                       = {};

      // Initialize timestepping
      const unsigned int time_integration_order =
        TimeIntegration::get_scheme_order(
          params.time_integration_data.interation_scheme);

      this->dts.assign(time_integration_order, 0);
      this->dts[0] = params.time_integration_data.time_step_init;


      InitialRefine global_refine;
      if (params.geometry_data.global_refinement == "None")
        global_refine = InitialRefine::None;
      else if (params.geometry_data.global_refinement == "Base")
        global_refine = InitialRefine::Base;
      else if (params.geometry_data.global_refinement == "Full")
        global_refine = InitialRefine::Full;
      else
        AssertThrow(false, ExcNotImplemented());

      const unsigned int n_refinements_remaining = create_grid(global_refine);
      this->n_global_levels_0 =
        tria.n_global_levels() + n_refinements_remaining;

      initialize();

      const auto initialize_solution =
        [&](std::vector<typename VectorType::BlockType *> solution_ptr,
            MyTimerOutput &                               timer) {
          MyScope scope(timer, "initialize_solution");

          AssertThrow(initial_solution->n_components() <= solution_ptr.size(),
                      ExcMessage(
                        "Number of initial values components (" +
                        std::to_string(initial_solution->n_components()) +
                        ") exceeds the number of blocks provided (" +
                        std::to_string(solution_ptr.size()) + ")."));

          for (unsigned int c = 0; c < initial_solution->n_components(); ++c)
            {
              initial_solution->set_component(c);

              VectorTools::interpolate(mapping,
                                       dof_handler,
                                       *initial_solution,
                                       *solution_ptr[c]);

              constraints.distribute(*solution_ptr[c]);
              solution_ptr[c]->zero_out_ghost_values();
            }
        };

      run(initial_solution->n_components(), initialize_solution);
    }

    Problem(const Parameters &params, const std::string &restart_path)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , mapping(1)
      , quad(QIterated<1>(QGauss<1>(params.approximation_data.n_points_1D),
                          params.approximation_data.n_subdivisions))
      , dof_handler(tria)
    {
      MyScope("Problem::constructor");

      fe = create_fe(params.approximation_data.fe_degree,
                     params.approximation_data.n_subdivisions);

      // 0) load internal state
      unsigned int n_ranks;
      unsigned int n_initial_components;
      unsigned int n_blocks_per_vector;
      unsigned int n_blocks_total;
      unsigned int n_integration_order;
      bool         flexible_output;
      bool         full_history;

      std::ifstream                   in_stream(restart_path + "_driver");
      boost::archive::binary_iarchive fisb(in_stream);
      fisb >> flexible_output;
      fisb >> full_history;
      fisb >> n_ranks;

      const auto n_mpi_processes =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

      AssertThrow(
        flexible_output || (n_ranks == n_mpi_processes),
        ExcMessage(
          "You are not using flexible serialization. You can restart this simulation only with " +
          std::to_string(n_ranks) + "! But you have " +
          std::to_string(n_mpi_processes) + "!"));

      fisb >> n_initial_components;
      fisb >> n_blocks_per_vector;
      fisb >> n_blocks_total;

      const unsigned int time_integration_order =
        TimeIntegration::get_scheme_order(
          params.time_integration_data.interation_scheme);

      if (full_history)
        {
          fisb >> n_integration_order;
          this->dts.resize(n_integration_order);
        }
      else
        {
          this->dts.resize(time_integration_order);
        }

      // Read the rest
      fisb >> *this;

      // Check if the data structures are consistent
      if (full_history)
        {
          // Strictly speaking, the number of vectors should be equal to the
          // time integration order + 1, however, we skipped the recent old
          // solution since it will get overwritten anyway, so we neither save
          // it not load during restarts.
          AssertDimension(this->dts.size(),
                          n_blocks_total / n_blocks_per_vector);

          // We do resize anyways since the user might have changed the
          // integration scheme
          this->dts.resize(time_integration_order);
        }
      else
        {
          std::fill(std::next(this->dts.begin()), this->dts.end(), 0.);
        }

      // 1) create coarse mesh
      create_grid(InitialRefine::None);

      // 2) load mesh refinement (incl. vectors)
      tria.load(restart_path + "_tria");

      // note: for flexible restart file, do not repartition here,
      // since else the attached vectors will be lost; in the case of
      // non-flexible restart filed, we need to repartition here, since
      // Triangulation::load() does not guarantee that the saved and loaded
      // files are identical
      if (params.restart_data.flexible_output == false)
        tria.repartition();

      // 3) initialize data structures
      initialize();

      // 4) helper function to initialize solution vector
      const auto initialize_solution =
        [&, flexible_output, n_blocks_total, restart_path](
          std::vector<typename VectorType::BlockType *> solution_ptr,
          MyTimerOutput &                               timer) {
          MyScope scope(timer, "deserialize_solution");

          if (n_blocks_total < solution_ptr.size())
            solution_ptr.resize(n_blocks_total);

          if (flexible_output)
            {
              parallel::distributed::
                SolutionTransfer<dim, typename VectorType::BlockType>
                  solution_transfer(dof_handler);

              // In order to perform deserialization, we need to add dummy
              // vectors which will get deleted at the end
              std::vector<std::shared_ptr<typename VectorType::BlockType>>
                dummy_vecs;
              if (n_blocks_total > solution_ptr.size())
                for (unsigned int i = solution_ptr.size(); i < n_blocks_total;
                     ++i)
                  {
                    auto dummy =
                      std::make_shared<typename VectorType::BlockType>(
                        *solution_ptr[0]);
                    dummy_vecs.push_back(dummy);
                    solution_ptr.push_back(dummy.get());
                  }

              solution_transfer.deserialize(solution_ptr);
            }
          else
            {
              parallel::distributed::
                SolutionSerialization<dim, typename VectorType::BlockType>
                  solution_serialization(dof_handler);

              solution_serialization.add_vectors(solution_ptr);

              solution_serialization.load(restart_path + "_vectors");
            }
        };

      // 5) run time loop
      run(n_initial_components, initialize_solution);
    }

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int /*version*/)
    {
      // geometry
      ar &geometry_domain_boundaries;
      ar &geometry_r_max;
      ar &geometry_interface_width;

      // counters
      ar &n_global_levels_0;
      ar &time_last_output;
      ar &n_timestep;
      ar &n_linear_iterations;
      ar &n_non_linear_iterations;
      ar &n_residual_evaluations;
      ar &n_failed_tries;
      ar &n_failed_linear_iterations;
      ar &n_failed_non_linear_iterations;
      ar &n_failed_residual_evaluations;
      ar &max_reached_dt;
      ar &restart_counter;
      ar &t;
      ar &counters;
      ar &boost::serialization::make_array(dts.data(), dts.size());
    }

    unsigned int
    create_grid(InitialRefine global_refine)
    {
      MyScope("Problem::create_grid");

      std::pair<dealii::Point<dim>, dealii::Point<dim>> boundaries;

      if (!params.geometry_data.custom_bounding_box)
        {
          boundaries  = geometry_domain_boundaries;
          double rmax = geometry_r_max;

          for (unsigned int i = 0; i < dim; i++)
            {
              boundaries.first[i] -=
                params.geometry_data.boundary_factor * rmax;
              boundaries.second[i] +=
                params.geometry_data.boundary_factor * rmax;
            }
        }
      else
        {
          if (dim >= 1)
            {
              boundaries.first[0] =
                params.geometry_data.bounding_box_data.x_min;
              boundaries.second[0] =
                params.geometry_data.bounding_box_data.x_max;
            }
          if (dim >= 2)
            {
              boundaries.first[1] =
                params.geometry_data.bounding_box_data.y_min;
              boundaries.second[1] =
                params.geometry_data.bounding_box_data.y_max;
            }
          if (dim == 3)
            {
              boundaries.first[2] =
                params.geometry_data.bounding_box_data.z_min;
              boundaries.second[2] =
                params.geometry_data.bounding_box_data.z_max;
            }
        }

      const unsigned int n_refinements_remaining =
        create_mesh(tria,
                    boundaries.first,
                    boundaries.second,
                    geometry_interface_width,
                    params.geometry_data.divisions_per_interface,
                    params.geometry_data.periodic,
                    global_refine,
                    params.geometry_data.max_prime,
                    params.geometry_data.max_level0_divisions_per_interface,
                    params.approximation_data.n_subdivisions);

      helper = std::make_unique<dealii::parallel::Helper<dim>>(tria);

      const auto weight_function = parallel::hanging_nodes_weighting<dim>(
        *helper, params.geometry_data.hanging_node_weight);
      tria.signals.weight.connect(weight_function);

      tria.repartition();

      return n_refinements_remaining;
    }

    void
    initialize(const unsigned int n_components = 0)
    {
      // setup DoFHandlers, ...
      if (true)
        {
          MyScope("Problem::initialize::dofhandler");
          dof_handler.distribute_dofs(*fe);
        }

      // ... constraints, and ...
      if (true)
        {
          MyScope("Problem::initialize::constraints");
          constraints.clear();
          constraints.reinit(
            DoFTools::extract_locally_relevant_dofs(dof_handler));
          DoFTools::make_hanging_node_constraints(dof_handler, constraints);

          if (params.geometry_data.periodic)
            {
              std::vector<GridTools::PeriodicFacePair<
                typename dealii::DoFHandler<dim>::cell_iterator>>
                periodicity_vector;

              for (unsigned int d = 0; d < dim; ++d)
                {
                  GridTools::collect_periodic_faces(
                    dof_handler, 2 * d, 2 * d + 1, d, periodicity_vector);
                }

              DoFTools::make_periodicity_constraints<dim, dim>(
                periodicity_vector, constraints);
            }

          constraints.close();
        }

      // ... MatrixFree
      if (true)
        {
          MyScope("Problem::initialize::matrix_free");
          typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
            additional_data;
          additional_data.mapping_update_flags =
            update_values | update_gradients;

          if (params.advection_data.enable || true)
            //(!params.output_data.domain_integrals.empty() &&
            // params.output_data.use_control_box))
            additional_data.mapping_update_flags |= update_quadrature_points;

          matrix_free.reinit(
            mapping, dof_handler, constraints, quad, additional_data);
        }

      if ((params.preconditioners_data.outer_preconditioner == "GMG") ||
          (params.preconditioners_data.outer_preconditioner == "BlockGMG") ||
          ((params.preconditioners_data.outer_preconditioner ==
            "BlockPreconditioner2") &&
           ((params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "GMG") ||
            (params.preconditioners_data.block_preconditioner_2_data
               .block_1_preconditioner == "BlockGMG"))))
        {
          MyScope("Problem::initialize::multigrid");

          mg_triangulations = MGTransferGlobalCoarseningTools::
            create_geometric_coarsening_sequence(tria);

          const unsigned int min_level = 0;
          const unsigned int max_level = mg_triangulations.size() - 1;

          const unsigned int max_max_level =
            n_global_levels_0 + params.adaptivity_data.max_refinement_depth;

          mg_dof_handlers.resize(min_level, max_level);
          transfers.resize(min_level, max_level);

          if (min_level != mg_constraints.min_level() ||
              max_max_level != mg_constraints.max_level())
            mg_constraints.resize(min_level, max_max_level);

          if (min_level != mg_matrix_free.min_level() ||
              max_max_level != mg_matrix_free.max_level())
            mg_matrix_free.resize(min_level, max_max_level);

          for (unsigned int l = min_level; l <= max_level; ++l)
            {
              auto &dof_handler = mg_dof_handlers[l];
              auto &constraints = mg_constraints[l];
              auto &matrix_free = mg_matrix_free[l];

              dof_handler.reinit(*mg_triangulations[l]);
              dof_handler.distribute_dofs(*fe);

              constraints.clear();
              constraints.reinit(
                DoFTools::extract_locally_relevant_dofs(dof_handler));
              DoFTools::make_hanging_node_constraints(dof_handler, constraints);
              Assert(params.geometry_data.periodic == false,
                     ExcNotImplemented());
              constraints.close();

              typename MatrixFree<dim, Number, VectorizedArrayType>::
                AdditionalData additional_data;
              additional_data.mapping_update_flags =
                update_values | update_gradients;

              matrix_free.reinit(
                mapping, dof_handler, constraints, quad, additional_data);
            }

          for (auto l = min_level; l < max_level; ++l)
            transfers[l + 1].reinit(mg_dof_handlers[l + 1],
                                    mg_dof_handlers[l],
                                    mg_constraints[l + 1],
                                    mg_constraints[l]);

          transfer = std::make_shared<
            MGTransferGlobalCoarsening<dim, typename VectorType::BlockType>>(
            transfers, [&](const auto l, auto &vec) {
              mg_matrix_free[l].initialize_dof_vector(vec);
            });
        }

      types::global_cell_index n_cells_w_hn  = 0;
      types::global_cell_index n_cells_wo_hn = 0;

      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            if (helper->is_constrained(cell))
              n_cells_w_hn++;
            else
              n_cells_wo_hn++;
          }

      const auto min_max_avg_n_cells_w_hn =
        Utilities::MPI::min_max_avg(n_cells_w_hn, MPI_COMM_WORLD);
      const auto min_max_avg_n_cells_wo_hn =
        Utilities::MPI::min_max_avg(n_cells_wo_hn, MPI_COMM_WORLD);
      const auto min_max_avg_n_cells =
        Utilities::MPI::min_max_avg(n_cells_w_hn + n_cells_wo_hn,
                                    MPI_COMM_WORLD);

      AssertDimension(
        static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.sum) +
          static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.sum),
        tria.n_global_active_cells());

      // clang-format off
      pcout_statistics << "System statistics:" << std::endl;
      pcout_statistics << "  - n cell:                    " << static_cast<types::global_cell_index>(min_max_avg_n_cells.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n cell w hanging nodes:    " << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_w_hn.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n cell wo hanging nodes:   " << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.sum) 
                                                            << " (min: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.min) 
                                                            << ", max: " 
                                                            << static_cast<types::global_cell_index>(min_max_avg_n_cells_wo_hn.max) 
                                                            << ")" << std::endl;
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;

      if(transfer)
      {
      pcout_statistics << "  - n cells on levels:         ";
      for(const auto & tria : mg_triangulations)
        pcout_statistics << tria->n_global_active_cells () << " ";
      pcout_statistics << std::endl;
      }

      pcout_statistics << "  - n dofs:                    " << dof_handler.n_dofs() << std::endl;
      if(n_components > 0)
        pcout_statistics << "  - n components:              " << n_components << std::endl;
      pcout_statistics << std::endl;
      // clang-format on
    }

    void
    run(const unsigned int n_initial_components,
        const std::function<
          void(std::vector<typename VectorType::BlockType *> solution_ptr,
               MyTimerOutput &)> &initialize_solution)
    {
      TimerPredicate restart_predicate(params.restart_data.type,
                                       params.restart_data.type == "n_calls" ?
                                         n_timestep :
                                         t,
                                       params.restart_data.interval);

      const unsigned int time_integration_order = dts.size();

      // Compute energy and mobility parameters
      double                            A, B, kappa_c, kappa_p;
      std::shared_ptr<MobilityProvider> mobility_provider;
      if (params.material_data.type == "Abstract")
        {
          A       = params.material_data.energy_abstract_data.A;
          B       = params.material_data.energy_abstract_data.B;
          kappa_c = params.material_data.energy_abstract_data.kappa_c;
          kappa_p = params.material_data.energy_abstract_data.kappa_p;

          mobility_provider = std::make_shared<ProviderAbstract>(
            params.material_data.mobility_abstract_data.Mvol,
            params.material_data.mobility_abstract_data.Mvap,
            params.material_data.mobility_abstract_data.Msurf,
            params.material_data.mobility_abstract_data.Mgb,
            params.material_data.mobility_abstract_data.L);
        }
      else
        {
          const double gamma_surf =
            params.material_data.energy_realistic_data.surface_energy;
          const double gamma_gb =
            params.material_data.energy_realistic_data.grain_boundary_energy;

          auto energy_params =
            compute_energy_params(gamma_surf,
                                  gamma_gb,
                                  geometry_interface_width,
                                  params.material_data.length_scale,
                                  params.material_data.energy_scale);

          A       = energy_params.A;
          B       = energy_params.B;
          kappa_c = energy_params.kappa_c;
          kappa_p = energy_params.kappa_p;

          const auto temperature_function =
            std::make_shared<Function1DPiecewise<double>>(
              params.material_data.temperature);

          mobility_provider = std::make_shared<ProviderRealistic>(
            params.material_data.mobility_realistic_data.omega,
            params.material_data.mobility_realistic_data.D_vol0,
            params.material_data.mobility_realistic_data.D_vap0,
            params.material_data.mobility_realistic_data.D_surf0,
            params.material_data.mobility_realistic_data.D_gb0,
            params.material_data.mobility_realistic_data.Q_vol,
            params.material_data.mobility_realistic_data.Q_vap,
            params.material_data.mobility_realistic_data.Q_surf,
            params.material_data.mobility_realistic_data.Q_gb,
            params.material_data.mobility_realistic_data.D_gb_mob0,
            params.material_data.mobility_realistic_data.Q_gb_mob,
            geometry_interface_width,
            params.material_data.time_scale,
            params.material_data.length_scale,
            params.material_data.energy_scale,
            temperature_function);

          const auto mobility_reference = mobility_provider->calculate(0.0);

          // Output material data
          pcout << "Effective model material data: " << std::endl;

          pcout << "- energy parameters:" << std::endl;
          pcout << "  A       = " << A << std::endl;
          pcout << "  B       = " << B << std::endl;
          pcout << "  kappa_c = " << kappa_c << std::endl;
          pcout << "  kappa_p = " << kappa_p << std::endl;

          pcout << "- mobility parameters:" << std::endl;
          pcout << "  Mvol    = " << mobility_reference.Mvol << std::endl;
          pcout << "  Mvap    = " << mobility_reference.Mvap << std::endl;
          pcout << "  Msurf   = " << mobility_reference.Msurf << std::endl;
          pcout << "  Mgb     = " << mobility_reference.Mgb << std::endl;
          pcout << "  L       = " << mobility_reference.L << std::endl;

          pcout << std::endl;
        }

      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        A, B, kappa_c, kappa_p, mobility_provider, time_integration_order);

      pcout << "Mobility type: "
            << (sintering_data.use_tensorial_mobility ? "tensorial" : "scalar")
            << std::endl;
      pcout << std::endl;

      sintering_data.set_n_components(n_initial_components);

      // Reference to the current timestep for convinience
      auto &dt = dts[0];

      TimeIntegration::SolutionHistory<VectorType> solution_history(
        time_integration_order + 1);

      MGLevelObject<SinteringOperatorData<dim, VectorizedArrayType>>
        mg_sintering_data(0,
                          n_global_levels_0 +
                            params.adaptivity_data.max_refinement_depth,
                          sintering_data);

      // New grains can not appear in current sintering simulations
      const bool         allow_new_grains        = false;
      const unsigned int order_parameters_offset = 2;

      GrainTracker::Tracker<dim, Number> grain_tracker(
        dof_handler,
        tria,
        !params.geometry_data.minimize_order_parameters,
        allow_new_grains,
        params.grain_tracker_data.fast_reassignment,
        MAX_SINTERING_GRAINS,
        params.grain_tracker_data.threshold_lower,
        params.grain_tracker_data.threshold_upper,
        params.grain_tracker_data.buffer_distance_ratio,
        params.grain_tracker_data.buffer_distance_fixed,
        order_parameters_offset);

      // Advection physics for shrinkage
      AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism(
        params.advection_data.enable,
        params.advection_data.mt,
        params.advection_data.mr,
        grain_tracker);

      auto nonlinear_operator = create_sintering_operator<dim,
                                                          Number,
                                                          VectorizedArrayType,
                                                          VectorType,
                                                          NonLinearOperator>(
        matrix_free,
        constraints,
        sintering_data,
        solution_history,
        advection_mechanism,
        params.matrix_based,
        params.material_data.mechanics_data.E,
        params.material_data.mechanics_data.nu);

      std::unique_ptr<NonLinearSolvers::JacobianBase<Number>> jacobian_operator;

      if (params.nonlinear_data.jacobi_free == false)
        jacobian_operator = std::make_unique<
          NonLinearSolvers::JacobianWrapper<Number, NonLinearOperator>>(
          nonlinear_operator);
      else
        jacobian_operator = std::make_unique<
          NonLinearSolvers::JacobianFree<Number, NonLinearOperator>>(
          nonlinear_operator);

      // Save all blocks at quadrature points if either the advection mechanism
      // is enabled or the coupled diffusion based sintering operator is used
      const bool save_all_blocks =
        params.advection_data.enable ||
        std::is_same_v<
          SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>,
          NonLinearOperator>;

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (transfer)
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          sintering_data,
          matrix_free,
          constraints,
          mg_sintering_data,
          mg_matrix_free,
          mg_constraints,
          transfer,
          params.preconditioners_data.block_preconditioner_2_data);
      else if (params.preconditioners_data.outer_preconditioner ==
               "BlockPreconditioner2")
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          sintering_data,
          matrix_free,
          constraints,
          params.preconditioners_data.block_preconditioner_2_data);
      else
        preconditioner = Preconditioners::create(
          nonlinear_operator, params.preconditioners_data.outer_preconditioner);

      // ... linear solver
      std::unique_ptr<SolverControl> solver_control_l;
      if (true) // TODO: make parameter
        {
          solver_control_l =
            std::make_unique<ReductionControl>(params.nonlinear_data.l_max_iter,
                                               params.nonlinear_data.l_abs_tol,
                                               params.nonlinear_data.l_rel_tol);
        }
      else
        {
          solver_control_l = std::make_unique<IterationNumberControl>(10);
        }

      // Enable tracking residual evolution
      if (params.nonlinear_data.verbosity >= 2) // TODO
        solver_control_l->enable_history_data();

      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (params.nonlinear_data.l_solver == "GMRES")
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearSolvers::JacobianBase<Number>,
          Preconditioners::PreconditionerBase<Number>>>(
          *jacobian_operator,
          *preconditioner,
          *solver_control_l,
          params.nonlinear_data.gmres_data);
      else if (params.nonlinear_data.l_solver == "Relaxation")
        linear_solver = std::make_unique<LinearSolvers::SolverRelaxation<
          NonLinearSolvers::JacobianBase<Number>,
          Preconditioners::PreconditionerBase<Number>>>(*jacobian_operator,
                                                        *preconditioner);
      else if (params.nonlinear_data.l_solver == "IDR")
        linear_solver = std::make_unique<LinearSolvers::SolverIDRWrapper<
          NonLinearSolvers::JacobianBase<Number>,
          Preconditioners::PreconditionerBase<Number>>>(*jacobian_operator,
                                                        *preconditioner,
                                                        *solver_control_l);
      else if (params.nonlinear_data.l_solver == "Bicgstab")
        linear_solver = std::make_unique<LinearSolvers::SolverBicgstabWrapper<
          NonLinearSolvers::JacobianBase<Number>,
          Preconditioners::PreconditionerBase<Number>>>(
          *jacobian_operator,
          *preconditioner,
          *solver_control_l,
          params.nonlinear_data.l_bisgstab_tries);

      MyTimerOutput timer;
      TimerCollection::configure(params.profiling_data.output_time_interval);

      // Define vector to store additional initializers for additional vectors
      std::vector<std::function<void()>> additional_initializations;

      // Initialization of data for advanced postprocessing (if needed)
      PostprocOperator<dim, Number, VectorizedArrayType> postproc_operator(
        matrix_free, constraints, sintering_data, params.matrix_based);

      MassMatrix<dim, Number, VectorizedArrayType> mass_operator(matrix_free,
                                                                 constraints);

      std::unique_ptr<ReductionControl> postproc_solver_control_l;
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        postproc_preconditioner;
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>>
        postproc_linear_solver;

      VectorType postproc_lhs, postproc_rhs;

      if (params.output_data.fluxes_divergences)
        {
          postproc_solver_control_l =
            std::make_unique<ReductionControl>(params.nonlinear_data.l_max_iter,
                                               params.nonlinear_data.l_abs_tol,
                                               params.nonlinear_data.l_rel_tol);

          postproc_preconditioner =
            Preconditioners::create(mass_operator, "InverseDiagonalMatrix");

          postproc_linear_solver =
            std::make_unique<LinearSolvers::SolverGMRESWrapper<
              MassMatrix<dim, Number, VectorizedArrayType>,
              Preconditioners::PreconditionerBase<Number>>>(
              mass_operator,
              *postproc_preconditioner,
              *postproc_solver_control_l,
              params.nonlinear_data.gmres_data);

          additional_initializations.emplace_back(
            [&postproc_operator, &postproc_lhs, &postproc_rhs]() {
              postproc_operator.initialize_dof_vector(postproc_lhs);
              postproc_operator.initialize_dof_vector(postproc_rhs);
            });
        }

      AdvectionOperator<dim, Number, VectorizedArrayType> advection_operator(
        params.advection_data.k,
        params.advection_data.cgb,
        params.advection_data.ceq,
        matrix_free,
        constraints,
        sintering_data,
        grain_tracker);

      // ... non-linear Newton solver
      NonLinearSolvers::NewtonSolverSolverControl statistics(
        params.nonlinear_data.nl_max_iter,
        params.nonlinear_data.nl_abs_tol,
        params.nonlinear_data.nl_rel_tol);

      // Lambda to compute residual
      const auto nl_residual = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::residual");

        // Compute forces
        if (params.advection_data.enable)
          advection_operator.evaluate_forces(src, advection_mechanism);

        nonlinear_operator.evaluate_nonlinear_residual(dst, src);

        statistics.increment_residual_evaluations(1);
      };

      // Lambda to set up jacobian
      const auto nl_setup_jacobian = [&](const auto &current_u) {
        MyScope scope(timer, "time_loop::newton::setup_jacobian");

        sintering_data.fill_quadrature_point_values(
          matrix_free,
          current_u,
          params.advection_data.enable,
          save_all_blocks);

        // TODO disable this feature for a while, fix later
        // nonlinear_operator.update_state(current_u);

        nonlinear_operator.do_update();

        if (params.nonlinear_data.fdm_jacobian_approximation)
          {
            AssertThrow(params.matrix_based, ExcNotImplemented());

            const double epsilon   = 1e-7;
            const double tolerance = 1e-12;

            auto &system_matrix = nonlinear_operator.get_system_matrix();

            const unsigned int n_blocks = current_u.n_blocks();

            // for (unsigned int b = 0; b < n_blocks; ++b)
            //  for (unsigned int i = 0; i < current_u.block(b).size();
            //  ++i)
            //    if(constraints.is_constrained (i))
            //                system_matrix.set(b + i * n_blocks,
            //                                  b + i * n_blocks,
            //                                  1.0);

            system_matrix = 0.0;

            VectorType src, dst, dst_;
            src.reinit(current_u);
            dst.reinit(current_u);
            dst_.reinit(current_u);

            src.copy_locally_owned_data_from(current_u);

            nl_residual(src, dst_);

            const auto locally_owned_dofs = dof_handler.locally_owned_dofs();

            for (unsigned int b = 0; b < n_blocks; ++b)
              for (unsigned int i = 0; i < current_u.block(b).size(); ++i)
                {
                  if (locally_owned_dofs.is_element(i))
                    src.block(b)[i] += epsilon;

                  nl_residual(src, dst);

                  if (locally_owned_dofs.is_element(i))
                    src.block(b)[i] -= epsilon;

                  for (unsigned int b_ = 0; b_ < n_blocks; ++b_)
                    for (unsigned int i_ = 0; i_ < current_u.block(b).size();
                         ++i_)
                      if (locally_owned_dofs.is_element(i_))
                        {
                          if (nonlinear_operator.get_sparsity_pattern().exists(
                                b_ + i_ * n_blocks, b + i * n_blocks))
                            {
                              const Number value =
                                (dst.block(b_)[i_] - dst_.block(b_)[i_]) /
                                epsilon;

                              if (std::abs(value) > tolerance)
                                system_matrix.set(b_ + i_ * n_blocks,
                                                  b + i * n_blocks,
                                                  value);
                              else if ((b == b_) && (i == i_))
                                system_matrix.set(b_ + i_ * n_blocks,
                                                  b + i * n_blocks,
                                                  1.0);
                            }
                        }
                }
          }

        jacobian_operator->reinit(current_u);
      };

      // Lambda to update preconditioner
      const auto nl_setup_preconditioner = [&](const auto &current_u) {
        MyScope scope(timer, "time_loop::newton::setup_preconditioner");

        if (transfer) // update multigrid levels
          {
            const unsigned int min_level = transfers.min_level();
            const unsigned int max_level = transfers.max_level();
            const unsigned int n_blocks  = current_u.n_blocks();

            for (unsigned int l = min_level; l <= max_level; ++l)
              mg_sintering_data[l].time_data.set_all_dt(dts);

            MGLevelObject<VectorType> mg_current_u(min_level, max_level);

            // acitve level
            mg_current_u[max_level].reinit(n_blocks);
            for (unsigned int b = 0; b < n_blocks; ++b)
              mg_matrix_free[max_level].initialize_dof_vector(
                mg_current_u[max_level].block(b));
            mg_current_u[max_level].copy_locally_owned_data_from(current_u);

            // coarser levels
            for (unsigned int l = max_level; l > min_level; --l)
              {
                mg_current_u[l - 1].reinit(n_blocks);
                for (unsigned int b = 0; b < n_blocks; ++b)
                  mg_matrix_free[l - 1].initialize_dof_vector(
                    mg_current_u[l - 1].block(b));

                for (unsigned int b = 0; b < n_blocks; ++b)
                  transfers[l].interpolate(mg_current_u[l - 1].block(b),
                                           mg_current_u[l].block(b));
              }

            for (unsigned int l = min_level; l <= max_level; ++l)
              {
                mg_sintering_data[l].set_n_components(
                  sintering_data.n_components());
                mg_sintering_data[l].fill_quadrature_point_values(
                  mg_matrix_free[l],
                  mg_current_u[l],
                  params.advection_data.enable,
                  save_all_blocks);
              }
          }

        preconditioner->do_update();
      };

      // Lambda to solve system using jacobian
      const auto nl_solve_with_jacobian = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::solve_with_jacobian");

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.set_zero(const_cast<VectorType &>(src).block(b));

        const unsigned int n_iterations = linear_solver->solve(dst, src);

        for (unsigned int b = 0; b < src.n_blocks(); ++b)
          constraints.distribute(dst.block(b));

        if (params.nonlinear_data.verbosity >= 2 &&
            !solver_control_l->get_history_data().empty())
          {
            pcout << " - l_res_abs: ";
            for (const auto res : solver_control_l->get_history_data())
              pcout << res << " ";
            pcout << std::endl;

            std::vector<double> &res_history =
              const_cast<std::vector<double> &>(
                solver_control_l->get_history_data());
            res_history.clear();
          }

        statistics.increment_linear_iterations(n_iterations);

        return n_iterations;
      };

      // Lambda to check iterations and output stats
      double check_value_0     = 0.0;
      double check_value_0_ch  = 0.0;
      double check_value_0_ac  = 0.0;
      double check_value_0_mec = 0.0;

      unsigned int previous_linear_iter = 0;

      const auto nl_check_iteration_status = [&](const auto  step,
                                                 const auto  check_value,
                                                 const auto &x,
                                                 const auto &r) {
        (void)x;

        double check_value_ch  = 0.0;
        double check_value_ac  = 0.0;
        double check_value_mec = 0.0;

        if (r.n_blocks() > 0)
          {
            check_value_ch =
              std::sqrt(r.block(0).norm_sqr() + r.block(0).norm_sqr());

            for (unsigned int b = 2; b < sintering_data.n_components(); ++b)
              check_value_ac += r.block(b).norm_sqr();
            check_value_ac = std::sqrt(check_value_ac);

            if (nonlinear_operator.n_components() >
                sintering_data.n_components())
              for (unsigned int b = r.n_blocks() - dim; b < r.n_blocks(); ++b)
                check_value_mec += r.block(b).norm_sqr();
            check_value_mec = std::sqrt(check_value_mec);
          }

        if (step == 0)
          {
            check_value_0     = check_value;
            check_value_0_ch  = check_value_ch;
            check_value_0_ac  = check_value_ac;
            check_value_0_mec = check_value_mec;

            previous_linear_iter = 0;
          }

        const unsigned int step_linear_iter =
          statistics.n_linear_iterations() - previous_linear_iter;

        previous_linear_iter = statistics.n_linear_iterations();

        if (pcout.is_active())
          {
            if (step == 0)
              printf(
                "\nit      res_abs      res_rel   ch_rel_abs   ch_res_rel   ac_rel_abs   ac_res_rel  mec_rel_abs  mec_res_rel  linear_iter\n");

            if (step == 0)
              printf(
                "%2d %.6e ------------ %.6e ------------ %.6e ------------ %.6e ------------ %12d\n",
                step,
                check_value,
                check_value_ch,
                check_value_ac,
                check_value_mec,
                step_linear_iter);
            else
              printf("%2d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %12d\n",
                     step,
                     check_value,
                     check_value_0 ? check_value / check_value_0 : 0.,
                     check_value_ch,
                     check_value_0_ch ? check_value_ch / check_value_0_ch : 0.,
                     check_value_ac,
                     check_value_0_ac ? check_value_ac / check_value_0_ac : 0.,
                     check_value_mec,
                     check_value_0_mec ? check_value_mec / check_value_0_mec :
                                         0.,
                     step_linear_iter);
          }

        /* This function does not really test anything and simply prints more
         * details on the residual evolution. We have different return status
         * here due to the fact that our DampedNewtonSolver::check() works
         * slightly differently in comparison to
         * NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR). The latter has
         * a bit strange not very obvious logic.
         */
        return params.nonlinear_data.nonlinear_solver_type == "NOX" ?
                 SolverControl::iterate :
                 SolverControl::success;
      };

      std::unique_ptr<NonLinearSolvers::NewtonSolver<VectorType>>
        non_linear_solver_executor;

      if (params.nonlinear_data.nonlinear_solver_type == "damped")
        {
          NonLinearSolvers::DampedNewtonSolver<VectorType> non_linear_solver(
            statistics,
            NonLinearSolvers::NewtonSolverAdditionalData(
              params.nonlinear_data.newton_do_update,
              params.nonlinear_data.newton_threshold_newton_iter,
              params.nonlinear_data.newton_threshold_linear_iter,
              params.nonlinear_data.newton_reuse_preconditioner,
              params.nonlinear_data.newton_use_damping));

          non_linear_solver.reinit_vector = [&](auto &vector) {
            MyScope scope(timer, "time_loop::newton::reinit_vector");

            nonlinear_operator.initialize_dof_vector(vector);
          };

          non_linear_solver.residual             = nl_residual;
          non_linear_solver.setup_jacobian       = nl_setup_jacobian;
          non_linear_solver.setup_preconditioner = nl_setup_preconditioner;
          non_linear_solver.solve_with_jacobian  = nl_solve_with_jacobian;

          if (params.nonlinear_data.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

          non_linear_solver_executor =
            std::make_unique<NonLinearSolvers::NonLinearSolverWrapper<
              VectorType,
              NonLinearSolvers::DampedNewtonSolver<VectorType>>>(
              std::move(non_linear_solver));
        }
      else if (params.nonlinear_data.nonlinear_solver_type == "NOX")
        {
          Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
            Teuchos::rcp(new Teuchos::ParameterList);

          non_linear_parameters->set("Nonlinear Solver", "Line Search Based");

          auto &printParams = non_linear_parameters->sublist("Printing");
          printParams.set("Output Information",
                          params.nonlinear_data.nox_data.output_information);

          auto &dir_parameters = non_linear_parameters->sublist("Direction");
          dir_parameters.set("Method",
                             params.nonlinear_data.nox_data.direction_method);

          auto &search_parameters =
            non_linear_parameters->sublist("Line Search");
          search_parameters.set(
            "Method", params.nonlinear_data.nox_data.line_search_method);

          // Params for polynomial
          auto &poly_params = search_parameters.sublist("Polynomial");
          poly_params.set(
            "Interpolation Type",
            params.nonlinear_data.nox_data.line_search_interpolation_type);

          typename TrilinosWrappers::NOXSolver<VectorType>::AdditionalData
            additional_data(params.nonlinear_data.nl_max_iter,
                            params.nonlinear_data.nl_abs_tol,
                            params.nonlinear_data.nl_rel_tol,
                            params.nonlinear_data.newton_threshold_newton_iter,
                            params.nonlinear_data.newton_threshold_linear_iter);

          TrilinosWrappers::NOXSolver<VectorType> non_linear_solver(
            additional_data, non_linear_parameters);

          non_linear_solver.residual = [&nl_residual](const auto &src,
                                                      auto &      dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [&nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          non_linear_solver.setup_preconditioner =
            [&nl_setup_preconditioner](const auto &current_u) {
              nl_setup_preconditioner(current_u);
              return 0;
            };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [&nl_solve_with_jacobian](const auto & src,
                                      auto &       dst,
                                      const double tolerance) {
              (void)tolerance;
              return nl_solve_with_jacobian(src, dst);
            };

          non_linear_solver.apply_jacobian =
            [&jacobian_operator](const auto &src, auto &dst) {
              jacobian_operator->vmult(dst, src);
              return 0;
            };

          if (params.nonlinear_data.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

          non_linear_solver_executor =
            std::make_unique<NonLinearSolvers::NonLinearSolverWrapper<
              VectorType,
              TrilinosWrappers::NOXSolver<VectorType>>>(
              std::move(non_linear_solver), statistics);
        }
#if defined(DEAL_II_WITH_PETSC) && defined(USE_SNES)
      else if (params.nonlinear_data.nonlinear_solver_type == "SNES")
        {
          typename NonLinearSolvers::SNESSolver<VectorType>::AdditionalData
            additional_data(params.nonlinear_data.nl_max_iter,
                            params.nonlinear_data.nl_abs_tol,
                            params.nonlinear_data.nl_rel_tol,
                            params.nonlinear_data.newton_threshold_newton_iter,
                            params.nonlinear_data.newton_threshold_linear_iter,
                            params.nonlinear_data.snes_data.line_search_name);

          NonLinearSolvers::SNESSolver<VectorType> non_linear_solver(
            additional_data, params.nonlinear_data.snes_data.solver_name);

          non_linear_solver.residual = [&nl_residual](const auto &src,
                                                      auto &      dst) {
            nl_residual(src, dst);
            return 0;
          };

          non_linear_solver.setup_jacobian =
            [&nl_setup_jacobian](const auto &current_u) {
              nl_setup_jacobian(current_u);
              return 0;
            };

          non_linear_solver.setup_preconditioner =
            [&nl_setup_preconditioner](const auto &current_u) {
              nl_setup_preconditioner(current_u);
              return 0;
            };

          non_linear_solver.solve_with_jacobian_and_track_n_linear_iterations =
            [&nl_solve_with_jacobian](const auto & src,
                                      auto &       dst,
                                      const double tolerance) {
              (void)tolerance;
              return nl_solve_with_jacobian(src, dst);
            };

          non_linear_solver.apply_jacobian =
            [&jacobian_operator](const auto &src, auto &dst) {
              jacobian_operator->vmult(dst, src);
              return 0;
            };

          if (params.nonlinear_data.verbosity >= 1) // TODO
            non_linear_solver.check_iteration_status =
              nl_check_iteration_status;

          non_linear_solver_executor =
            std::make_unique<NonLinearSolvers::NonLinearSolverWrapper<
              VectorType,
              NonLinearSolvers::SNESSolver<VectorType>>>(
              std::move(non_linear_solver), statistics);
        }
#endif
      else
        AssertThrow(false, ExcNotImplemented());

      // set initial condition

      std::function<void(VectorType &)> f_init =
        [&nonlinear_operator](VectorType &v) {
          nonlinear_operator.initialize_dof_vector(v);
        };

      solution_history.apply(f_init);
      std::for_each(additional_initializations.begin(),
                    additional_initializations.end(),
                    [](auto &a_init) { a_init(); });

      VectorType &solution = solution_history.get_current_solution();

      bool system_has_changed = true;

      const auto execute_coarsening_and_refinement =
        [&](const double t,
            const double top_fraction_of_cells,
            const double bottom_fraction_of_cells) {
          MyScope scope(timer, "execute_coarsening_and_refinement");

          pcout << "Execute refinement/coarsening:" << std::endl;

          system_has_changed = true;

          auto solutions_except_recent = solution_history.filter(true, false);
          auto old_old_solutions       = solution_history.filter(false, false);

          output_result(
            solution, nonlinear_operator, grain_tracker, t, "refinement");

          // 1) copy solution so that it has the right ghosting
          const auto partitioner =
            std::make_shared<Utilities::MPI::Partitioner>(
              dof_handler.locally_owned_dofs(),
              DoFTools::extract_locally_relevant_dofs(dof_handler),
              dof_handler.get_communicator());

          VectorType solution_dealii(solutions_except_recent.n_blocks_total());
          for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
            solution_dealii.block(b).reinit(partitioner);

          solutions_except_recent.flatten(solution_dealii);

          for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
            constraints.distribute(solution_dealii.block(b));

          solution_dealii.update_ghost_values();

          // 2) estimate errors
          Vector<float> estimated_error_per_cell(tria.n_active_cells());

          for (unsigned int b = 2; b < sintering_data.n_components(); ++b)
            {
              Vector<float> estimated_error_per_cell_temp(
                tria.n_active_cells());

              KellyErrorEstimator<dim>::estimate(
                this->dof_handler,
                Quadrature<dim - 1>(quad),
                std::map<types::boundary_id, const Function<dim> *>(),
                solution_dealii.block(b),
                estimated_error_per_cell_temp,
                {},
                nullptr,
                0,
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

              for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
                estimated_error_per_cell[i] +=
                  estimated_error_per_cell_temp[i] *
                  estimated_error_per_cell_temp[i];
            }

          for (unsigned int i = 0; i < estimated_error_per_cell.size(); ++i)
            estimated_error_per_cell[i] =
              std::sqrt(estimated_error_per_cell[i]);

          // 3) mark automatically cells for coarsening/refinement, ...
          parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_fraction(tria,
                                              estimated_error_per_cell,
                                              top_fraction_of_cells,
                                              bottom_fraction_of_cells);

          // make sure that cells close to the interfaces are refined, ...
          Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());
          const Number   val_min = 0.05;
          const Number   val_max = 0.95;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (cell->is_locally_owned() == false || cell->refine_flag_set())
                continue;

              for (unsigned int b = 2; b < sintering_data.n_components(); ++b)
                {
                  cell->get_dof_values(solution_dealii.block(b), values);

                  Number val_avg = 0;

                  for (unsigned int i = 0; i < values.size(); ++i)
                    {
                      val_avg += values[i];

                      if (val_min < values[i] && values[i] < val_max)
                        {
                          cell->clear_coarsen_flag();
                          cell->set_refine_flag();

                          break;
                        }
                    }

                  if (!cell->refine_flag_set())
                    {
                      // In case if a cell has values, e.g., close to 0 or 1
                      val_avg /= values.size();
                      if (val_min < val_avg && val_avg < val_max)
                        {
                          cell->clear_coarsen_flag();
                          cell->set_refine_flag();
                        }
                    }

                  if (cell->refine_flag_set())
                    break;
                }
            }

          // and limit the number of levels
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->refine_flag_set() &&
                (static_cast<unsigned int>(cell->level()) ==
                 ((this->n_global_levels_0 - 1) +
                  params.adaptivity_data.max_refinement_depth)))
              cell->clear_refine_flag();
            else if (cell->coarsen_flag_set() &&
                     (static_cast<unsigned int>(cell->level()) ==
                      ((this->n_global_levels_0 - 1) -
                       std::min((this->n_global_levels_0 - 1),
                                params.adaptivity_data.min_refinement_depth))))
              cell->clear_coarsen_flag();

          // 4) perform interpolation and initialize data structures
          tria.prepare_coarsening_and_refinement();

          parallel::distributed::
            SolutionTransfer<dim, typename VectorType::BlockType>
              solution_trans(dof_handler);

          std::vector<const typename VectorType::BlockType *>
            solution_dealii_ptr(solution_dealii.n_blocks());
          for (unsigned int b = 0; b < solution_dealii.n_blocks(); ++b)
            solution_dealii_ptr[b] = &solution_dealii.block(b);

          solution_trans.prepare_for_coarsening_and_refinement(
            solution_dealii_ptr);

          tria.execute_coarsening_and_refinement();

          initialize(solution.n_blocks());

          nonlinear_operator.clear();
          non_linear_solver_executor->clear();
          preconditioner->clear();

          mass_operator.clear();
          if (params.output_data.fluxes_divergences)
            postproc_preconditioner->clear();

          solutions_except_recent.apply(f_init);
          std::for_each(additional_initializations.begin(),
                        additional_initializations.end(),
                        [](auto &a_init) { a_init(); });

          auto solution_ptr = solutions_except_recent.get_all_blocks();

          solution_trans.interpolate(solution_ptr);

          // note: apply constraints since the Newton solver expects this
          for (unsigned int b = 0; b < solution_ptr.size(); ++b)
            constraints.distribute(*solution_ptr[b]);

          old_old_solutions.update_ghost_values();

          output_result(
            solution, nonlinear_operator, grain_tracker, t, "refinement");

          if (params.output_data.mesh_overhead_estimate)
            Postprocessors::estimate_overhead(mapping, dof_handler, solution);
        };

      const auto run_grain_tracker = [&](const double t,
                                         const bool   do_initialize = false) {
        MyScope scope(timer, "time_loop::grain_tracker");

        pcout << "Execute grain tracker:" << std::endl;

        system_has_changed = true;

        auto solutions_except_recent = solution_history.filter(true, false);
        auto old_old_solutions       = solution_history.filter(false, false);

        solutions_except_recent.update_ghost_values();


        const auto time_total = std::chrono::system_clock::now();

        std::tuple<bool, bool> gt_status;
        if (do_initialize)
          {
            MyScope scope(timer, "time_loop::grain_tracker::initial_setup");
            gt_status =
              grain_tracker.initial_setup(solution, sintering_data.n_grains());
          }
        else
          {
            MyScope scope(timer, "time_loop::grain_tracker::track");
            gt_status =
              grain_tracker.track(solution, sintering_data.n_grains());
          }

        const bool has_reassigned_grains = std::get<0>(gt_status);
        const bool has_op_number_changed = std::get<1>(gt_status);

        pcout << std::boolalpha
              << "has_reassigned_grains = " << has_reassigned_grains << " | "
              << "has_op_number_changed = " << has_op_number_changed
              << std::noboolalpha << std::endl;

        const double time_total_double =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - time_total)
            .count() /
          1e9;

        pcout << "Grain tracker CPU time = "
              << Utilities::MPI::max(time_total_double, MPI_COMM_WORLD)
              << " sec" << std::endl;

        grain_tracker.print_current_grains(pcout);

        // Rebuild data structures if grains have been reassigned
        if (has_reassigned_grains || has_op_number_changed)
          {
            output_result(
              solution, nonlinear_operator, grain_tracker, t, "remap");

            if (has_op_number_changed)
              {
                const unsigned int n_grains_new =
                  grain_tracker.get_active_order_parameters().size();
                const unsigned int n_components_old = solution.n_blocks();
                const unsigned int n_components_new =
                  n_components_old - sintering_data.n_grains() + n_grains_new;

                pcout << "\033[34mChanging number of components from "
                      << n_components_old << " to " << n_components_new
                      << "\033[0m" << std::endl;

                AssertThrow(n_grains_new <= MAX_SINTERING_GRAINS,
                            ExcMessage("Number of grains (" +
                                       std::to_string(n_grains_new) +
                                       ") exceeds the maximum value (" +
                                       std::to_string(MAX_SINTERING_GRAINS) +
                                       ")."));

                nonlinear_operator.clear();
                non_linear_solver_executor->clear();
                preconditioner->clear();

                /**
                 * If the number of components has reduced, then we remap first
                 * and then alter the number of blocks in the solution vector.
                 * If the number of components has increased, then we need to
                 * add new blocks to the solution vector prior to remapping.
                 */

                const unsigned int distance =
                  nonlinear_operator.n_components() -
                  sintering_data.n_components();

                // Perform remapping
                auto all_solution_vectors =
                  solutions_except_recent.get_all_solutions();

                if (has_reassigned_grains &&
                    n_components_new < n_components_old)
                  {
                    MyScope scope(timer, "time_loop::grain_tracker::remap");

                    grain_tracker.remap(all_solution_vectors);

                    // Move the to be deleted components to the end
                    if (distance > 0)
                      for (auto &sol : all_solution_vectors)
                        for (unsigned int i = sintering_data.n_components() - 1;
                             i >= n_components_new;
                             --i)
                          sol->move_block(i, i + distance);
                  }

                // Resize solution vector (delete or add blocks)
                solutions_except_recent.apply(
                  [&](auto &sol) { sol.reinit(n_components_new); });

                if (has_reassigned_grains &&
                    n_components_new > n_components_old)
                  {
                    MyScope scope(timer, "time_loop::grain_tracker::remap");

                    // Move the newly created before displacements
                    if (distance > 0)
                      for (auto &sol : all_solution_vectors)
                        for (unsigned int i = nonlinear_operator.n_components();
                             i < n_components_new;
                             ++i)
                          sol->move_block(i, i - distance);

                    grain_tracker.remap(all_solution_vectors);
                  }

                // Change number of components after remapping completed
                sintering_data.set_n_components(n_grains_new + 2);
              }
            else if (has_reassigned_grains)
              {
                MyScope scope(timer, "time_loop::grain_tracker::remap");

                // Perform remapping
                auto all_solution_vectors =
                  solutions_except_recent.get_all_solutions();

                grain_tracker.remap(all_solution_vectors);
              }

            // We need to call track again if advection mechanism is used
            // in order to keep op_particle_ids in sync
            if (params.advection_data.enable)
              {
                MyScope scope(timer, "time_loop::grain_tracker::track_fast");

                const bool skip_reassignment = false;
                grain_tracker.track(solution,
                                    sintering_data.n_grains(),
                                    skip_reassignment);
              }

            output_result(
              solution, nonlinear_operator, grain_tracker, t, "remap");
          }

        pcout << std::endl;

        solution.zero_out_ghost_values();
        old_old_solutions.update_ghost_values();
      };

      // Impose boundary conditions
      const auto impose_boundary_conditions = [&](const double t) {
        (void)t;

        // Update mechanical constraints for the coupled model
        // Currently only fixing the central section along x-axis
        if constexpr (std::is_base_of_v<
                        SinteringOperatorCoupledBase<dim,
                                                     Number,
                                                     VectorizedArrayType,
                                                     NonLinearOperator>,
                        NonLinearOperator>)
          {
            MyScope scope(timer, "impose_boundary_conditions");

            pcout << "Impose boundary conditions:" << std::endl;
            pcout << "  - type: " << params.boundary_conditions.type
                  << std::endl;
            pcout << "  - direction: " << params.boundary_conditions.direction
                  << std::endl;

            auto &displ_constraints_indices =
              nonlinear_operator.get_zero_constraints_indices();

            if (params.boundary_conditions.type == "CentralSection")
              {
                clamp_central_section<dim>(
                  displ_constraints_indices,
                  matrix_free,
                  solution.block(0),
                  params.boundary_conditions.direction);
              }
            else if (params.boundary_conditions.type == "CentralParticle")
              {
                AssertThrow(params.grain_tracker_data.grain_tracker_frequency >
                              0,
                            ExcMessage("Grain tracker has to be enabled"));

                Point<dim> origin = find_center_origin(
                  matrix_free.get_dof_handler().get_triangulation(),
                  grain_tracker);
                pcout << "  - origin for clamping the section: " << origin
                      << std::endl;

                clamp_section_within_particle<dim>(
                  displ_constraints_indices,
                  matrix_free,
                  sintering_data,
                  grain_tracker,
                  solution,
                  origin,
                  params.boundary_conditions.direction);
              }
            else if (params.boundary_conditions.type == "Domain")
              {
                clamp_domain<dim>(displ_constraints_indices, matrix_free);
              }

            pcout << std::endl;
          }
      };

      // Initialize all solutions except the evry old one, it will get
      // overwritten anyway
      initialize_solution(
        solution_history.filter(true, false, true).get_all_blocks(), timer);

      // initial local refinement
      if (t == 0.0 && (params.adaptivity_data.refinement_frequency > 0 ||
                       params.geometry_data.global_refinement != "Full"))
        {
          // Initialize only the current solution
          const auto solution_ptr =
            solution_history.filter(true, false, false).get_all_blocks();

          // If global_refinement is not Full, then we need use more agressive
          // refinement strategy here
          double top_fraction_of_cells =
            params.adaptivity_data.top_fraction_of_cells;
          const double bottom_fraction_of_cells =
            params.adaptivity_data.bottom_fraction_of_cells;
          if (params.geometry_data.global_refinement == "None")
            top_fraction_of_cells *= 3;
          else if (params.geometry_data.global_refinement == "Base")
            top_fraction_of_cells *= 2.5;

          top_fraction_of_cells = std::min(top_fraction_of_cells, 1.0);

          const unsigned int n_init_refinements =
            std::max(std::min(tria.n_global_levels() - 1,
                              params.adaptivity_data.min_refinement_depth),
                     this->n_global_levels_0 - tria.n_global_levels() +
                       params.adaptivity_data.max_refinement_depth);

          pcout << "Number of local refinements to be performed: "
                << n_init_refinements << std::endl;

          for (unsigned int i = 0; i < n_init_refinements; ++i)
            {
              execute_coarsening_and_refinement(t,
                                                top_fraction_of_cells,
                                                bottom_fraction_of_cells);
              initialize_solution(solution_ptr, timer);
            }
        }

      // Grain tracker - first run after we have initial configuration defined
      if (params.grain_tracker_data.grain_tracker_frequency > 0)
        run_grain_tracker(t, /*do_initialize = */ true);

      // Impose boundary conditions
      impose_boundary_conditions(t);

      // Build additional output
      std::function<void(DataOut<dim> & data_out)> additional_output;

      if (params.output_data.fluxes_divergences)
        additional_output =
          [&postproc_operator, &postproc_lhs, this](DataOut<dim> &data_out) {
            postproc_operator.add_data_vectors(data_out, postproc_lhs, {});
          };

      if (t == 0.0 && params.output_data.output_time_interval > 0.0)
        output_result(solution,
                      nonlinear_operator,
                      grain_tracker,
                      time_last_output,
                      "solution",
                      additional_output);

      // run time loop
      {
        while (t <= params.time_integration_data.time_end)
          {
            TimerOutput::Scope scope(timer(), "time_loop");

            // Perform sanity check
            if (params.time_integration_data.sanity_check_solution)
              nonlinear_operator.sanity_check(solution);

            bool do_mesh_refinement = false;
            bool do_grain_tracker   = false;
            if (n_timestep != 0)
              {
                // If quality control is enabled, then frequency is not used
                if (params.adaptivity_data.quality_control)
                  {
                    const auto only_order_parameters =
                      solution.create_view(2, sintering_data.n_components());

                    const auto quality =
                      Postprocessors::estimate_mesh_quality_min(
                        dof_handler, *only_order_parameters);

                    do_mesh_refinement =
                      quality < params.adaptivity_data.quality_min;
                  }
                else
                  {
                    do_mesh_refinement =
                      params.adaptivity_data.refinement_frequency > 0 &&
                      n_timestep %
                          params.adaptivity_data.refinement_frequency ==
                        0;
                  }

                // If advection is enabled, then execute grain tracker
                if (params.advection_data.enable)
                  do_grain_tracker = true;
                // If mesh quality control is enabled and grain tracker is asked
                // to run at the same time, then execute it synchronously
                else if (params.adaptivity_data.quality_control &&
                         params.grain_tracker_data.track_with_quality)
                  do_grain_tracker = do_mesh_refinement;
                // Otherwise use the default frequency settings
                else
                  do_grain_tracker =
                    params.grain_tracker_data.grain_tracker_frequency > 0 &&
                    n_timestep %
                        params.grain_tracker_data.grain_tracker_frequency ==
                      0;
              }

            if (do_mesh_refinement)
              execute_coarsening_and_refinement(
                t,
                params.adaptivity_data.top_fraction_of_cells,
                params.adaptivity_data.bottom_fraction_of_cells);

            if (do_grain_tracker)
              {
                try
                  {
                    run_grain_tracker(t, /*do_initialize = */ false);
                  }
                catch (const GrainTracker::ExcGrainsInconsistency &ex)
                  {
                    output_result(solution,
                                  nonlinear_operator,
                                  grain_tracker,
                                  time_last_output,
                                  "grains_inconsistency");

                    grain_tracker.print_old_grains(pcout);

                    AssertThrow(false, ExcMessage(ex.what()));
                  }
              }

            // Impose boundary conditions
            if (do_mesh_refinement)
              impose_boundary_conditions(t);

            // Set timesteps in order to update weights
            sintering_data.time_data.set_all_dt(dts);

            // Update material properties
            sintering_data.set_time(t);

            // Try to extrapolate initial guess
            if (params.time_integration_data.predictor != "None" && t > 0)
              {
                VectorType extrap;
                nonlinear_operator.initialize_dof_vector(extrap);

                if (params.time_integration_data.predictor == "Euler")
                  {
                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      extrap, solution);
                    extrap.sadd(-dt, solution);
                  }
                else if (params.time_integration_data.predictor == "Midpoint")
                  {
                    VectorType midpoint;
                    nonlinear_operator.initialize_dof_vector(midpoint);

                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      midpoint, solution);
                    midpoint.sadd(-dt / 2., solution);

                    nonlinear_operator.template evaluate_nonlinear_residual<0>(
                      extrap, midpoint);
                    extrap.sadd(-dt, solution);
                  }
                else if (params.time_integration_data.predictor == "Linear" &&
                         dts.size() > 1)
                  {
                    const double fac = dts[0] / dts[1];
                    solution_history.extrapolate(extrap, fac);
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }

                solution_history.set_recent_old_solution(solution);

                // Sanity check of the predicted value
                if (params.time_integration_data.sanity_check_predictor)
                  nonlinear_operator.sanity_check(extrap);

                solution = extrap;
              }
            else
              {
                solution_history.set_recent_old_solution(solution);
              }

            if (params.profiling_data.run_vmults && system_has_changed)
              {
                MyScope scope(timer, "time_loop::profiling_vmult");

                const bool old_timing_state =
                  nonlinear_operator.set_timing(false);

                sintering_data.fill_quadrature_point_values(
                  matrix_free,
                  solution,
                  params.advection_data.enable,
                  save_all_blocks);

                VectorType dst, src;

                nonlinear_operator.initialize_dof_vector(dst);
                nonlinear_operator.initialize_dof_vector(src);

                const unsigned int n_repetitions = 100;

                TimerOutput timer(pcout_statistics,
                                  TimerOutput::never,
                                  TimerOutput::wall_times);

                if (true)
                  {
                    TimerOutput::Scope scope(timer, "vmult_matrixfree");

                    for (unsigned int i = 0; i < n_repetitions; ++i)
                      nonlinear_operator.vmult(dst, src);
                  }

                if (true)
                  {
                    TimerOutput::Scope scope(timer, "vmult_helmholtz");

                    HelmholtzOperator<dim, Number, VectorizedArrayType>
                      helmholtz_operator(matrix_free, constraints, 1);

                    for (unsigned int i = 0; i < n_repetitions; ++i)
                      for (unsigned int b = 0; b < src.n_blocks(); ++b)
                        helmholtz_operator.vmult(dst.block(b), src.block(b));
                  }

                if (true)
                  {
                    TimerOutput::Scope scope(timer, "vmult_vector_helmholtz");

                    HelmholtzOperator<dim, Number, VectorizedArrayType>
                      helmholtz_operator(matrix_free,
                                         constraints,
                                         src.n_blocks());

                    for (unsigned int i = 0; i < n_repetitions; ++i)
                      helmholtz_operator.vmult(dst, src);
                  }

                if (true)
                  {
                    {
                      TimerOutput::Scope scope(timer,
                                               "vmult_matrixbased_assembly");

                      nonlinear_operator.get_system_matrix();
                    }

                    const auto &matrix = nonlinear_operator.get_system_matrix();

                    typename VectorType::BlockType src_, dst_;

                    const auto partitioner =
                      nonlinear_operator.get_system_partitioner();

                    src_.reinit(partitioner);
                    dst_.reinit(partitioner);

                    VectorTools::merge_components_fast(src, src_);

                    {
                      TimerOutput::Scope scope(timer, "vmult_matrixbased");

                      for (unsigned int i = 0; i < n_repetitions; ++i)
                        matrix.vmult(dst_, src_);
                    }

                    VectorTools::split_up_components_fast(dst_, dst);

                    if (params.matrix_based == false)
                      nonlinear_operator.clear_system_matrix();
                  }

                timer.print_wall_time_statistics(MPI_COMM_WORLD);

                nonlinear_operator.set_timing(old_timing_state);
              }

            bool has_converged = false;

            try
              {
                MyScope scope(timer, "time_loop::newton");

                // Reset statistics
                statistics.clear();

                // note: input/output (solution) needs/has the right
                // constraints applied
                non_linear_solver_executor->solve(solution);

                has_converged = true;

                pcout << std::endl;
                pcout << "t = " << t << ", t_n = " << n_timestep
                      << ", dt = " << dt << ":"
                      << " solved in " << statistics.n_newton_iterations()
                      << " Newton iterations and "
                      << statistics.n_linear_iterations()
                      << " linear iterations" << std::endl;

                n_timestep += 1;
                n_linear_iterations += statistics.n_linear_iterations();
                n_non_linear_iterations += statistics.n_newton_iterations();
                n_residual_evaluations += statistics.n_residual_evaluations();
                max_reached_dt = std::max(max_reached_dt, dt);

                // Commit current timestep
                t += dt;
                for (int i = dts.size() - 2; i >= 0; --i)
                  dts[i + 1] = dts[i];

                if (std::abs(t - params.time_integration_data.time_end) > 1e-9)
                  {
                    if (statistics.n_newton_iterations() <
                          params.time_integration_data
                            .desirable_newton_iterations &&
                        statistics.n_linear_iterations() <
                          params.time_integration_data
                            .desirable_linear_iterations)
                      {
                        dt *= params.time_integration_data.growth_factor;
                        pcout << "\033[32mIncreasing timestep, dt = " << dt
                              << "\033[0m" << std::endl;

                        if (dt > params.time_integration_data.time_step_max)
                          {
                            dt = params.time_integration_data.time_step_max;
                          }
                      }

                    if (t + dt > params.time_integration_data.time_end)
                      {
                        dt = params.time_integration_data.time_end - t;
                      }
                  }

                if (params.profiling_data.output_memory_consumption &&
                    system_has_changed)
                  {
                    // general-purpose: tria, dofhander, constraints
                    const auto mc_tria =
                      Utilities::MPI::sum<double>(tria.memory_consumption(),
                                                  MPI_COMM_WORLD);
                    const auto mc_dofhandler = Utilities::MPI::sum<double>(
                      dof_handler.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_affine_constraints =
                      Utilities::MPI::sum<double>(
                        constraints.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_matrix_free = Utilities::MPI::sum<double>(
                      matrix_free.memory_consumption(), MPI_COMM_WORLD);

                    // sintering-related data
                    const auto mc_sintering_data = Utilities::MPI::sum<double>(
                      sintering_data.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_nonlinear_operator =
                      Utilities::MPI::sum<double>(
                        nonlinear_operator.memory_consumption(),
                        MPI_COMM_WORLD);
                    const auto mc_solution_history =
                      Utilities::MPI::sum<double>(
                        solution_history.memory_consumption(), MPI_COMM_WORLD);
                    const auto mc_preconditioner = Utilities::MPI::sum<double>(
                      preconditioner->memory_consumption(), MPI_COMM_WORLD);

                    const auto mc_total =
                      mc_tria + mc_dofhandler + mc_affine_constraints +
                      mc_matrix_free + mc_nonlinear_operator +
                      mc_solution_history + mc_preconditioner;

                    pcout << "Memory consumption:     " << mc_total / 1e9
                          << std::endl;
                    pcout << " - triangulation:       " << mc_tria / 1e9
                          << std::endl;
                    pcout << " - dof-handler:         " << mc_dofhandler / 1e9
                          << std::endl;
                    pcout << " - affine constraints:  "
                          << mc_affine_constraints / 1e9 << std::endl;
                    pcout << " - matrix-free:         " << mc_matrix_free / 1e9
                          << std::endl;
                    pcout << " - sintering data:      "
                          << mc_sintering_data / 1e9 << std::endl;
                    pcout << " - non-linear operator: "
                          << mc_nonlinear_operator / 1e9 << std::endl;
                    pcout << " - vectors:             "
                          << mc_solution_history / 1e9 << std::endl;
                    pcout << " - preconditioner:      "
                          << mc_preconditioner / 1e9 << std::endl;
                  }

                // Posptrocessing to calculate divergences of fluxes
                if (params.output_data.fluxes_divergences)
                  {
                    MyScope scope(timer,
                                  "time_loop::newton::fluxes_divergences");

                    postproc_preconditioner->do_update();

                    postproc_operator.evaluate_rhs(postproc_rhs, solution);

                    for (unsigned int b = 0; b < postproc_lhs.n_blocks(); ++b)
                      {
                        const auto view_lhs =
                          postproc_lhs.create_view(b, b + 1);
                        const auto view_rhs =
                          postproc_rhs.create_view(b, b + 1);
                        postproc_linear_solver->solve(*view_lhs, *view_rhs);
                      }
                  }

                // Print grain forces
                if (params.advection_data.enable)
                  advection_mechanism.print_forces(pcout);
              }
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &e)
              {
                dt *= 0.5;
                pcout << "\033[31m" << e.message()
                      << " Reducing timestep, dt = " << dt << "\033[0m"
                      << std::endl;

                n_failed_tries += 1;
                n_failed_linear_iterations += statistics.n_linear_iterations();
                n_failed_non_linear_iterations +=
                  statistics.n_newton_iterations();
                n_failed_residual_evaluations +=
                  statistics.n_residual_evaluations();

                solution = solution_history.get_recent_old_solution();

                output_result(solution,
                              nonlinear_operator,
                              grain_tracker,
                              time_last_output,
                              "newton_not_converged");

                AssertThrow(
                  dt > params.time_integration_data.time_step_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }
            catch (const SolverControl::NoConvergence &)
              {
                dt *= 0.5;
                pcout
                  << "\033[33mLinear solver did not converge, reducing timestep, dt = "
                  << dt << "\033[0m" << std::endl;

                n_failed_tries += 1;
                n_failed_linear_iterations += statistics.n_linear_iterations();
                n_failed_non_linear_iterations +=
                  statistics.n_newton_iterations();
                n_failed_residual_evaluations +=
                  statistics.n_residual_evaluations();

                solution = solution_history.get_recent_old_solution();

                output_result(solution,
                              nonlinear_operator,
                              grain_tracker,
                              time_last_output,
                              "linear_solver_not_converged");

                AssertThrow(
                  dt > params.time_integration_data.time_step_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }

            if (has_converged)
              solution_history.commit_old_solutions();

            const bool is_last_time_step =
              has_converged &&
              (std::abs(t - params.time_integration_data.time_end) < 1e-8);

            if ((params.output_data.output_time_interval > 0.0) &&
                has_converged &&
                (t >
                   params.output_data.output_time_interval + time_last_output ||
                 is_last_time_step))
              {
                time_last_output = t;
                output_result(solution,
                              nonlinear_operator,
                              grain_tracker,
                              time_last_output,
                              "solution",
                              additional_output);
              }

            if (has_converged &&
                (is_last_time_step || restart_predicate.now(t)))
              {
                unsigned int current_restart_count = restart_counter++;

                if (params.restart_data.max_output != 0)
                  current_restart_count =
                    current_restart_count % params.restart_data.max_output;

                const std::string prefix =
                  params.restart_data.prefix + "_" +
                  std::to_string(current_restart_count);

                std::vector<const typename VectorType::BlockType *>
                  solution_ptr;

                if (params.restart_data.full_history)
                  {
                    auto all_except_old =
                      solution_history.filter(true, false, true);

                    const auto history_all_blocks =
                      all_except_old.get_all_blocks();
                    solution_ptr.assign(history_all_blocks.begin(),
                                        history_all_blocks.end());
                  }
                else
                  {
                    solution_ptr.resize(solution.n_blocks());
                    for (unsigned int b = 0; b < solution.n_blocks(); ++b)
                      solution_ptr[b] = &solution.block(b);
                  }

                if (params.restart_data.flexible_output)
                  {
                    if (!solution.has_ghost_elements())
                      solution.update_ghost_values();

                    parallel::distributed::
                      SolutionTransfer<dim, typename VectorType::BlockType>
                        solution_transfer(dof_handler);

                    solution_transfer.prepare_for_serialization(solution_ptr);
                    tria.save(prefix + "_tria");

                    solution.zero_out_ghost_values();
                  }
                else
                  {
                    parallel::distributed::
                      SolutionSerialization<dim, typename VectorType::BlockType>
                        solution_serialization(dof_handler);

                    solution_serialization.add_vectors(solution_ptr);

                    solution_serialization.save(prefix + "_vectors");
                    tria.save(prefix + "_tria");
                  }

                std::ofstream                   out_stream(prefix + "_driver");
                boost::archive::binary_oarchive fosb(out_stream);
                fosb << params.restart_data.flexible_output;
                fosb << params.restart_data.full_history;
                fosb << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
                fosb << sintering_data.n_components();
                fosb << solution.n_blocks();
                fosb << static_cast<unsigned int>(solution_ptr.size());

                if (params.restart_data.full_history)
                  fosb << static_cast<unsigned int>(dts.size());

                fosb << *this;
              }

            TimerCollection::print_all_wall_time_statistics();

            system_has_changed = false;
          }
      }

      // clang-format off
      pcout_statistics << std::endl;
      pcout_statistics << "Final statistics:" << std::endl;
      pcout_statistics << "  - n timesteps:               " << n_timestep << std::endl;
      pcout_statistics << "  - n non-linear iterations:   " << n_non_linear_iterations << std::endl;
      pcout_statistics << "  - n linear iterations:       " << n_linear_iterations << std::endl;
      pcout_statistics << "  - n residual evaluations:    " << n_residual_evaluations << std::endl;
      pcout_statistics << "  - avg non-linear iterations: " << static_cast<double>(n_non_linear_iterations) / n_timestep << std::endl;
      pcout_statistics << "  - avg linear iterations:     " << static_cast<double>(n_linear_iterations) / n_non_linear_iterations << std::endl;
      pcout_statistics << "  - max dt:                    " << max_reached_dt << std::endl;
      pcout_statistics << std::endl;
      pcout_statistics << "  - n failed tries:                   " << n_failed_tries << std::endl;
      pcout_statistics << "  - n failed non-linear iterations:   " << n_failed_non_linear_iterations << std::endl;
      pcout_statistics << "  - n failed linear iterations:       " << n_failed_linear_iterations << std::endl;
      pcout_statistics << "  - n failed residual evaluations:    " << n_failed_residual_evaluations << std::endl;      
      pcout_statistics << std::endl;
      // clang-format on

      TimerCollection::print_all_wall_time_statistics(true);
    }

  private:
    void
    output_result(
      const VectorType &                          solution,
      const NonLinearOperator &                   sintering_operator,
      const GrainTracker::Tracker<dim, Number> &  grain_tracker,
      const double                                t,
      const std::string                           label = "solution",
      std::function<void(DataOut<dim> &data_out)> additional_output = {})
    {
      if (!params.output_data.debug && label != "solution")
        return; // nothing to do for debug for non-solution

      if (counters.find(label) == counters.end())
        counters[label] = 0;

      // Create table handler
      TableHandler table;

      if (params.output_data.regular || label != "solution")
        {
          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells =
            params.output_data.higher_order_cells;

          DataOutWithRanges<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.set_flags(flags);

          if (params.output_data.fields.count("CH"))
            {
              data_out.add_data_vector(solution.block(0), "c");
              data_out.add_data_vector(solution.block(1), "mu");
            }

          if (params.output_data.fields.count("AC"))
            {
              for (unsigned int ig = 2;
                   ig < sintering_operator.get_data().n_components();
                   ++ig)
                data_out.add_data_vector(solution.block(ig),
                                         "eta" + std::to_string(ig - 2));
            }

          if (params.output_data.fields.count("displ") &&
              sintering_operator.n_components() >
                sintering_operator.get_data().n_components())
            {
              for (unsigned int b = solution.n_blocks() - dim;
                   b < solution.n_blocks();
                   ++b)
                data_out.add_data_vector(solution.block(b), "u");
            }

          sintering_operator.add_data_vectors(data_out,
                                              solution,
                                              params.output_data.fields);

          // Output additional data
          if (additional_output)
            additional_output(data_out);

          // Output subdomain structure
          if (params.output_data.fields.count("subdomain"))
            {
              Vector<float> subdomain(
                dof_handler.get_triangulation().n_active_cells());
              for (unsigned int i = 0; i < subdomain.size(); ++i)
                {
                  subdomain[i] =
                    dof_handler.get_triangulation().locally_owned_subdomain();
                }
              data_out.add_data_vector(subdomain, "subdomain");
            }

          data_out.build_patches(mapping, this->fe->tensor_degree());

          std::string output = params.output_data.vtk_path + "/" + label + "." +
                               std::to_string(counters[label]) + ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

          if (params.output_data.table)
            {
              table.add_value("step", counters[label]);
              table.add_value("time", t);
              table.add_value(
                "dt", sintering_operator.get_data().time_data.get_current_dt());

              if (true /*!params.output_data.domain_integrals.empty()*/)
                {
                  /*
                                    if (params.output_data.use_control_box)
                                      {
                                        Point<dim> bottom_left;
                                        Point<dim> top_right;

                                        if (dim >= 1)
                                          {
                                            bottom_left[0] =
                     params.output_data.control_box.x_min; top_right[0]   =
                     params.output_data.control_box.x_max;
                                          }

                                        if (dim >= 2)
                                          {
                                            bottom_left[1] =
                     params.output_data.control_box.y_min; top_right[1]   =
                     params.output_data.control_box.y_max;
                                          }

                                        if (dim >= 3)
                                          {
                                            bottom_left[2] =
                     params.output_data.control_box.z_min; top_right[2]   =
                     params.output_data.control_box.z_max;
                                          }

                                        BoundingBox control_box(bottom_left,
                     top_right);

                                        predicate = [&control_box](const
                     Point<dim> &p) { const auto zeros =
                     VectorizedArrayType(0.0); const auto ones =
                     VectorizedArrayType(1.0);

                                          VectorizedArrayType filter = ones;

                                          for(unsigned int d = 0; d < dim; ++d)
                                          {
                                            filter =
                     compare_and_apply_mask<SIMDComparison::greater_than>( p[d],
                     VectorizedArrayType(control_box.lower_bound[d]), filter,
                     zeros);

                                            filter =
                     compare_and_apply_mask<SIMDComparison::less_than>( p[d],
                     VectorizedArrayType(control_box.upper_bound[d]), filter,
                     zeros);
                                          }
                                        };
                                      }
                  */
                  std::set<std::string> domain_integrals{"solid_vol",
                                                         "surf_area",
                                                         "gb_area"};

                  std::vector<std::string> q_labels;
                  std::copy(domain_integrals.begin(),
                            domain_integrals.end(),
                            std::back_inserter(q_labels));

                  auto quantities =
                    Postprocessors::build_domain_quantities_evaluators<
                      dim,
                      VectorizedArrayType>(q_labels);

                  auto q_values =
                    sintering_operator.calc_domain_quantities(quantities,
                                                              solution);

                  for (unsigned int i = 0; i < quantities.size(); ++i)
                    table.add_value(q_labels[i], q_values[i]);
                }
            }
        }

      if (params.output_data.contours)
        {
          std::string output = params.output_data.vtk_path + "/contour_" +
                               label + "." + std::to_string(counters[label]) +
                               ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          Postprocessors::output_grain_contours(
            mapping,
            dof_handler,
            solution,
            0.5,
            output,
            sintering_operator.n_grains(),
            grain_tracker,
            params.output_data.n_coarsening_steps);
        }

      if (params.output_data.porosity)
        {
          std::string output = params.output_data.vtk_path + "/porosity_" +
                               label + "." + std::to_string(counters[label]) +
                               ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          Postprocessors::estimate_porosity(mapping,
                                            dof_handler,
                                            solution,
                                            output);
        }

      if (params.output_data.shrinkage)
        {
          std::string output = params.output_data.vtk_path + "/shrinkage_" +
                               label + "." + std::to_string(counters[label]) +
                               ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          const auto bb =
            Postprocessors::estimate_shrinkage(mapping, dof_handler, solution);

          Postprocessors::write_bounding_box(bb, mapping, dof_handler, output);

          if (params.output_data.table)
            {
              const std::vector labels = {"dim_x", "dim_y", "dim_z"};
              typename VectorType::value_type volume = 1.;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  const auto size = bb.side_length(d);
                  table.add_value(labels[d], size);
                  volume *= size;
                }
              table.add_value("volume", volume);
            }
        }

      if (params.output_data.quality)
        {
          std::string output = params.output_data.vtk_path + "/mesh_quality_" +
                               label + "." + std::to_string(counters[label]) +
                               ".vtu";

          pcout << "Outputing data at t = " << t << " (" << output << ")"
                << std::endl;

          const auto only_order_parameters =
            solution.create_view(2,
                                 sintering_operator.get_data().n_components());

          Postprocessors::estimate_mesh_quality(mapping,
                                                dof_handler,
                                                *only_order_parameters,
                                                output);
        }

      if (params.output_data.table)
        {
          std::string output =
            params.output_data.vtk_path + "/" + label + ".log";
          Postprocessors::write_table(table, t, MPI_COMM_WORLD, output);
        }

      counters[label]++;
    }


    std::shared_ptr<const FiniteElement<dim>>
    create_fe(const unsigned int fe_degree, const unsigned int n_subdivisions)
    {
      if (n_subdivisions == 1)
        {
          pcout << "Finite element: FE_Q<" << dim << ">, "
                << "n_subdivisions = " << n_subdivisions << std::endl;

          return std::make_shared<FE_Q<dim>>(fe_degree);
        }

      AssertThrow(fe_degree == 1,
                  ExcMessage(
                    "Either fe-degree or number of subdivisions has to be 1."));

      pcout << "Finite element: FE_Q_iso_Q1<" << dim << ">, "
            << "n_subdivisions = " << n_subdivisions << std::endl;

      return std::make_shared<FE_Q_iso_Q1<dim>>(n_subdivisions);
    }
  };
} // namespace Sintering
