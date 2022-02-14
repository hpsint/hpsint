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
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/vector_tools.h>

#include <pf-applications/sintering/operator.h>
#include <pf-applications/sintering/preconditioners.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim>
  class InitialValues : public dealii::Function<dim>
  {
  public:
    InitialValues(unsigned int n_components, double interface_offset = 0)
      : dealii::Function<dim>(n_components)
      , interface_offset(interface_offset)
    {}

    virtual std::pair<dealii::Point<dim>, dealii::Point<dim>>
    get_domain_boundaries() const = 0;

    virtual double
    get_r_max() const = 0;

    virtual double
    get_interface_width() const = 0;

  private:
    double interface_offset;

  protected:
    double
    is_in_sphere(const dealii::Point<dim> &point,
                 const dealii::Point<dim> &center,
                 double                    rc) const
    {
      double c = 0;

      double rm  = rc - interface_offset;
      double rad = center.distance(point);

      if (rad <= rm - get_interface_width() / 2.0)
        {
          c = 1;
        }
      else if (rad < rm + get_interface_width() / 2.0)
        {
          double outvalue = 0.;
          double invalue  = 1.;
          double int_pos =
            (rad - rm + get_interface_width() / 2.0) / get_interface_width();

          c = outvalue + (invalue - outvalue) *
                           (1.0 + std::cos(int_pos * numbers::PI)) / 2.0;
          // c = 0.5 - 0.5 * std::sin(numbers::PI * (rad - rm) /
          // get_interface_width());
        }

      return c;
    }
  };

  struct Parameters
  {
    unsigned int fe_degree   = 1;
    unsigned int n_points_1D = 2;

    double   top_fraction_of_cells    = 0.3;
    double   bottom_fraction_of_cells = 0.1;
    unsigned min_refinement_depth     = 3;
    unsigned max_refinement_depth     = 0;
    unsigned refinement_frequency     = 10;

    bool matrix_based = false;

    std::string outer_preconditioner = "BlockPreconditioner2";
    // std::string outer_preconditioner = "BlockPreconditioner3CH";
    // std::string outer_preconditioner = "ILU";

    BlockPreconditioner2Data   block_preconditioner_2_data;
    BlockPreconditioner3Data   block_preconditioner_3_data;
    BlockPreconditioner3CHData block_preconditioner_3_ch_data;

    bool print_time_loop = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
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
      const std::string preconditioner_types =
        "AMG|InverseBlockDiagonalMatrix|InverseDiagonalMatrix|ILU|InverseComponentBlockDiagonalMatrix";

      prm.add_parameter("FEDegree",
                        fe_degree,
                        "Degree of the shape the finite element.");
      prm.add_parameter("NPoints1D",
                        n_points_1D,
                        "Number of quadrature points.");
      prm.add_parameter(
        "OuterPreconditioner",
        outer_preconditioner,
        "Preconditioner to be used for the outer system.",
        Patterns::Selection(
          preconditioner_types +
          "|BlockPreconditioner2|BlockPreconditioner3|BlockPreconditioner3CH"));

      prm.enter_subsection("BlockPreconditioner2");
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_2_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block1Preconditioner",
                        block_preconditioner_2_data.block_1_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3");
      prm.add_parameter("Type",
                        block_preconditioner_3_data.type,
                        "Type of block preconditioner of CH system.",
                        Patterns::Selection("D|LD|RD|SYMM"));
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_3_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block0RelativeTolerance",
                        block_preconditioner_3_data.block_0_relative_tolerance,
                        "Relative tolerance of the first block.");
      prm.add_parameter("Block1Preconditioner",
                        block_preconditioner_3_data.block_1_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block1RelativeTolerance",
                        block_preconditioner_3_data.block_1_relative_tolerance,
                        "Relative tolerance of the second block.");
      prm.add_parameter("Block2Preconditioner",
                        block_preconditioner_3_data.block_2_preconditioner,
                        "Preconditioner to be used for the thrird block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2RelativeTolerance",
                        block_preconditioner_3_data.block_2_relative_tolerance,
                        "Relative tolerance of the third block.");
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3CH");
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_3_ch_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2Preconditioner",
                        block_preconditioner_3_ch_data.block_2_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();
    }
  };



  template <int dim,
            int n_grains,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class Problem
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // components number
    static constexpr unsigned int n_components = n_grains + 2;

    using NonLinearOperator =
      SinteringOperator<dim, Number, VectorizedArrayType>;

    // padding of computational domain
    static constexpr double boundary_factor = 0.5;

    // mesh
    static constexpr unsigned int elements_per_interface =
      8; // 4 - works well with AMR=off

    // time discretization
    static constexpr double t_end                = 100;
    static constexpr double dt_deseride          = 0.001;
    static constexpr double dt_max               = 1e3 * dt_deseride;
    static constexpr double dt_min               = 1e-2 * dt_deseride;
    static constexpr double dt_increment         = 1.2;
    static constexpr double output_time_interval = 10.0; // 0.0 means no output

    // desirable number of newton iterations
    static constexpr unsigned int desirable_newton_iterations = 5;
    static constexpr unsigned int desirable_linear_iterations = 100;

    //  model parameters
    static constexpr double A       = 16;
    static constexpr double B       = 1;
    static constexpr double Mvol    = 1e-2;
    static constexpr double Mvap    = 1e-10;
    static constexpr double Msurf   = 4;
    static constexpr double Mgb     = 0.4;
    static constexpr double L       = 1;
    static constexpr double kappa_c = 1;
    static constexpr double kappa_p = 0.5;

    const Parameters                          params;
    ConditionalOStream                        pcout;
    ConditionalOStream                        pcout_statistics;
    parallel::distributed::Triangulation<dim> tria;
    FESystem<dim>                             fe;
    MappingQ<dim>                             mapping;
    QGauss<dim>                               quad;
    DoFHandler<dim>                           dof_handler;
    DoFHandler<dim>                           dof_handler_ch;
    DoFHandler<dim>                           dof_handler_ac;
    DoFHandler<dim>                           dof_handler_scalar;

    AffineConstraints<Number> constraint;
    AffineConstraints<Number> constraint_ch;
    AffineConstraints<Number> constraint_ac;
    AffineConstraints<Number> constraint_scalar;

    const std::vector<const AffineConstraints<double> *> constraints;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    std::shared_ptr<InitialValues<dim>> initial_solution;

    Problem(const Parameters &                  params,
            std::shared_ptr<InitialValues<dim>> initial_solution)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , fe(FE_Q<dim>{params.fe_degree}, n_components)
      , mapping(1)
      , quad(params.n_points_1D)
      , dof_handler(tria)
      , dof_handler_ch(tria)
      , dof_handler_ac(tria)
      , dof_handler_scalar(tria)
      , constraints{&constraint,
                    &constraint_ch,
                    &constraint_ac,
                    &constraint_scalar}
      , initial_solution(initial_solution)
    {
      auto   boundaries = initial_solution->get_domain_boundaries();
      double rmax       = initial_solution->get_r_max();

      for (unsigned int i = 0; i < dim; i++)
        {
          boundaries.first[i] -= boundary_factor * rmax;
          boundaries.second[i] += boundary_factor * rmax;
        }

      create_mesh(tria,
                  boundaries.first,
                  boundaries.second,
                  initial_solution->get_interface_width(),
                  elements_per_interface);

      initialize();
    }

    void
    initialize()
    {
      // setup DoFHandlers, ...
      // a) complete system
      dof_handler.distribute_dofs(fe);
      // b) Cahn-Hilliard system
      dof_handler_ch.distribute_dofs(
        FESystem<dim>(FE_Q<dim>{params.fe_degree}, 2));
      // c) Allen-Cahn system
      dof_handler_ac.distribute_dofs(
        FESystem<dim>(FE_Q<dim>{params.fe_degree}, n_components - 2));
      // d) scalar
      dof_handler_scalar.distribute_dofs(FE_Q<dim>{params.fe_degree});

      // ... constraints, and ...
      constraint.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);
      constraint.close();

      constraint_ch.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_ch, constraint_ch);
      constraint_ch.close();

      constraint_ac.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_ac, constraint_ac);
      constraint_ac.close();

      constraint_scalar.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_scalar,
                                              constraint_scalar);
      constraint_scalar.close();

      // ... MatrixFree
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;
      // additional_data.use_fast_hanging_node_algorithm = false; // TODO

      const std::vector<const DoFHandler<dim> *> dof_handlers{
        &dof_handler, &dof_handler_ch, &dof_handler_ac, &dof_handler_scalar};

      matrix_free.reinit(
        mapping, dof_handlers, constraints, quad, additional_data);

      // clang-format off
      pcout_statistics << "System statistics:" << std::endl;
      pcout_statistics << "  - n cell:                    " << tria.n_global_active_cells() << std::endl;
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;
      pcout_statistics << "  - n dofs:                    " << dof_handler.n_dofs() << std::endl;
      pcout_statistics << std::endl;
      // clang-format on
    }

    void
    run()
    {
      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        A, B, Mvol, Mvap, Msurf, Mgb, L, kappa_c, kappa_p);

      // ... non-linear operator
      NonLinearOperator nonlinear_operator(matrix_free,
                                           constraints,
                                           sintering_data,
                                           params.matrix_based);

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (params.outer_preconditioner == "BlockPreconditioner2")
        preconditioner = std::make_unique<
          BlockPreconditioner2<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.block_preconditioner_2_data);
      else if (params.outer_preconditioner == "BlockPreconditioner3")
        preconditioner = std::make_unique<
          BlockPreconditioner3<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.block_preconditioner_3_data);
      else if (params.outer_preconditioner == "BlockPreconditioner3CH")
        preconditioner = std::make_unique<
          BlockPreconditioner3CH<dim, Number, VectorizedArrayType>>(
          nonlinear_operator,
          matrix_free,
          constraints,
          params.block_preconditioner_3_ch_data);
      else
        preconditioner = Preconditioners::create(nonlinear_operator,
                                                 params.outer_preconditioner);

      // ... linear solver
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (true)
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearOperator,
          Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                        *preconditioner);

      TimerOutput timer(pcout_statistics,
                        TimerOutput::never,
                        TimerOutput::wall_times);

      // ... non-linear Newton solver
      auto non_linear_solver =
        std::make_unique<NonLinearSolvers::NewtonSolver<VectorType>>();

      non_linear_solver->reinit_vector = [&](auto &vector) {
        MyScope scope(timer, "time_loop::newton::reinit_vector");

        nonlinear_operator.initialize_dof_vector(vector);
      };

      non_linear_solver->residual = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::residual");

        nonlinear_operator.evaluate_nonlinear_residual(dst, src);
      };

      non_linear_solver->setup_jacobian =
        [&](const auto &current_u, const bool do_update_preconditioner) {
          if (true)
            {
              MyScope scope(timer, "time_loop::newton::setup_jacobian");

              nonlinear_operator.evaluate_newton_step(current_u);
            }

          if (do_update_preconditioner)
            {
              MyScope scope(timer, "time_loop::newton::setup_preconditioner");
              preconditioner->do_update();
            }
        };

      non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::solve_with_jacobian");

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        constraint.set_zero(const_cast<VectorType &>(src));
        const unsigned int n_iterations = linear_solver->solve(dst, src);
        constraint.distribute(dst);

        return n_iterations;
      };


      // set initial condition
      VectorType solution;

      nonlinear_operator.initialize_dof_vector(solution);

      const auto initialize_solution = [&]() {
        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 *initial_solution,
                                 solution);
        solution.zero_out_ghost_values();
      };

      const unsigned int init_level = tria.n_global_levels() - 1;

      const auto execute_coarsening_and_refinement = [&]() {
        pcout << "Execute refinement/coarsening:" << std::endl;

        // 1) copy solution so that it has the right ghosting
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
        VectorType solution_dealii(dof_handler.locally_owned_dofs(),
                                   locally_relevant_dofs,
                                   dof_handler.get_communicator());

        // note: we do not need to apply constraints, since they are
        // are already set by the Newton solver
        solution_dealii.copy_locally_owned_data_from(solution);
        solution_dealii.update_ghost_values();

        // 2) estimate errors
        Vector<float> estimated_error_per_cell(tria.n_active_cells());

        std::vector<bool> mask(n_components, true);
        std::fill(mask.begin(), mask.begin() + 2, false);

        KellyErrorEstimator<dim>::estimate(
          this->dof_handler,
          QGauss<dim - 1>(this->dof_handler.get_fe().degree + 1),
          std::map<types::boundary_id, const Function<dim> *>(),
          solution_dealii,
          estimated_error_per_cell,
          mask,
          nullptr,
          0,
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

        // 3) mark automatically cells for coarsening/refinement, ...
        parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction(tria,
                                            estimated_error_per_cell,
                                            params.top_fraction_of_cells,
                                            params.bottom_fraction_of_cells);

        // make sure that cells close to the interfaces are refined, ...
        Vector<Number> values(fe.n_dofs_per_cell());
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            cell->get_dof_values(solution_dealii, values);

            for (unsigned int i = 0; i < values.size(); ++i)
              if (fe.system_to_component_index(i).first >= 2)
                if (0.05 < values[i] && values[i] < 0.95)
                  {
                    cell->clear_coarsen_flag();
                    cell->set_refine_flag();

                    break;
                  }
          }

        // and limit the number of levels
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->refine_flag_set() &&
              (static_cast<unsigned int>(cell->level()) ==
               (init_level + params.max_refinement_depth)))
            cell->clear_refine_flag();
          else if (cell->coarsen_flag_set() &&
                   (static_cast<unsigned int>(cell->level()) ==
                    (init_level -
                     std::min(init_level, params.min_refinement_depth))))
            cell->clear_coarsen_flag();

        // 4) perform interpolation and initialize data structures
        tria.prepare_coarsening_and_refinement();

        parallel::distributed::SolutionTransfer<dim, VectorType> solution_trans(
          dof_handler);
        solution_trans.prepare_for_coarsening_and_refinement(solution_dealii);

        tria.execute_coarsening_and_refinement();

        initialize();

        nonlinear_operator.clear();
        preconditioner->clear();

        VectorType interpolated_solution;
        nonlinear_operator.initialize_dof_vector(interpolated_solution);
        solution_trans.interpolate(interpolated_solution);

        nonlinear_operator.initialize_dof_vector(solution);
        solution.copy_locally_owned_data_from(interpolated_solution);

        // note: apply constraints since the Newton solver expects this
        constraint.distribute(solution);
      };

      initialize_solution();

      // initial local refinement
      if (params.refinement_frequency > 0)
        for (unsigned int i = 0; i < std::max(params.min_refinement_depth,
                                              params.max_refinement_depth);
             ++i)
          {
            execute_coarsening_and_refinement();
            initialize_solution();
          }

      double time_last_output = 0;

      if (output_time_interval > 0.0)
        output_result(solution, nonlinear_operator, time_last_output);

      unsigned int n_timestep              = 0;
      unsigned int n_linear_iterations     = 0;
      unsigned int n_non_linear_iterations = 0;
      double       max_reached_dt          = 0.0;

      // run time loop
      {
        TimerOutput::Scope scope(timer, "time_loop");
        for (double t = 0, dt = dt_deseride; t <= t_end;)
          {
            if (n_timestep != 0 && params.refinement_frequency > 0 &&
                n_timestep % params.refinement_frequency == 0)
              execute_coarsening_and_refinement();

            nonlinear_operator.set_timestep(dt);
            nonlinear_operator.set_previous_solution(solution);

            bool has_converged = false;

            try
              {
                MyScope scope(timer, "time_loop::newton");

                // note: input/output (solution) needs/has the right
                // constraints applied
                const auto statistics = non_linear_solver->solve(solution);

                has_converged = true;

                pcout << "t = " << t << ", dt = " << dt << ":"
                      << " solved in " << statistics.newton_iterations
                      << " Newton iterations and "
                      << statistics.linear_iterations << " linear iterations"
                      << std::endl;

                n_timestep += 1;
                n_linear_iterations += statistics.linear_iterations;
                n_non_linear_iterations += statistics.newton_iterations;
                max_reached_dt = std::max(max_reached_dt, dt);

                if (std::abs(t - t_end) > 1e-9)
                  {
                    if (statistics.newton_iterations <
                          desirable_newton_iterations &&
                        statistics.linear_iterations <
                          desirable_linear_iterations)
                      {
                        dt *= dt_increment;
                        pcout << "\033[32mIncreasing timestep, dt = " << dt
                              << "\033[0m" << std::endl;

                        if (dt > dt_max)
                          {
                            dt = dt_max;
                          }
                      }

                    if (t + dt > t_end)
                      {
                        dt = t_end - t;
                      }
                  }

                t += dt;
              }
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &)
              {
                dt *= 0.5;
                pcout
                  << "\033[31mNon-linear solver did not converge, reducing timestep, dt = "
                  << dt << "\033[0m" << std::endl;

                solution = nonlinear_operator.get_previous_solution();

                AssertThrow(
                  dt > dt_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }
            catch (const SolverControl::NoConvergence &)
              {
                dt *= 0.5;
                pcout
                  << "\033[33mLinear solver did not converge, reducing timestep, dt = "
                  << dt << "\033[0m" << std::endl;

                solution = nonlinear_operator.get_previous_solution();

                AssertThrow(
                  dt > dt_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }

            if ((output_time_interval > 0.0) && has_converged &&
                (t > output_time_interval + time_last_output))
              {
                time_last_output = t;
                output_result(solution, nonlinear_operator, time_last_output);
              }
          }
      }

      // clang-format off
      pcout_statistics << std::endl;
      pcout_statistics << "Final statistics:" << std::endl;
      pcout_statistics << "  - n timesteps:               " << n_timestep << std::endl;
      pcout_statistics << "  - n non-linear iterations:   " << n_non_linear_iterations << std::endl;
      pcout_statistics << "  - n linear iterations:       " << n_linear_iterations << std::endl;
      pcout_statistics << "  - avg non-linear iterations: " << static_cast<double>(n_non_linear_iterations) / n_timestep << std::endl;
      pcout_statistics << "  - avg linear iterations:     " << static_cast<double>(n_linear_iterations) / n_non_linear_iterations << std::endl;
      pcout_statistics << "  - max dt:                    " << max_reached_dt << std::endl;
      pcout_statistics << std::endl;
      // clang-format on

      timer.print_wall_time_statistics(MPI_COMM_WORLD);

      {
        nonlinear_operator.set_timestep(dt_deseride);
        nonlinear_operator.set_previous_solution(solution);
        nonlinear_operator.evaluate_newton_step(solution);

        VectorType dst, src;

        nonlinear_operator.initialize_dof_vector(dst);
        nonlinear_operator.initialize_dof_vector(src);

        const unsigned int n_repetitions = 1000;

        TimerOutput timer(pcout_statistics,
                          TimerOutput::never,
                          TimerOutput::wall_times);

        {
          TimerOutput::Scope scope(timer, "vmult_matrixfree");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            nonlinear_operator.vmult(dst, src);
        }

        {
          const auto &matrix = nonlinear_operator.get_system_matrix();

          TimerOutput::Scope scope(timer, "vmult_matrixbased");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            matrix.vmult(dst, src);
        }

        timer.print_wall_time_statistics(MPI_COMM_WORLD);
      }
    }

  private:
    void
    create_mesh(parallel::distributed::Triangulation<dim> &tria,
                const dealii::Point<dim> &                 bottom_left,
                const dealii::Point<dim> &                 top_right,
                const double                               interface_width,
                const unsigned int elements_per_interface)
    {
      const auto   domain_size   = top_right - bottom_left;
      const double domain_width  = domain_size[0];
      const double domain_height = domain_size[1];

      const unsigned int initial_ny = 10;
      const unsigned int initial_nx =
        static_cast<unsigned int>(domain_width / domain_height * initial_ny);

      const unsigned int n_refinements = static_cast<unsigned int>(
        std::round(std::log2(elements_per_interface / interface_width *
                             domain_height / initial_ny)));

      std::vector<unsigned int> subdivisions(dim);
      subdivisions[0] = initial_nx;
      subdivisions[1] = initial_ny;
      if (dim == 3)
        {
          const double       domain_depth = domain_size[2];
          const unsigned int initial_nz   = static_cast<unsigned int>(
            domain_depth / domain_height * initial_ny);
          subdivisions[2] = initial_nz;
        }

      dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                        subdivisions,
                                                        bottom_left,
                                                        top_right);

      if (n_refinements > 0)
        tria.refine_global(n_refinements);
    }

    void
    output_result(const VectorType &       solution,
                  const NonLinearOperator &sintering_operator,
                  const double             t)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      std::vector<std::string> names{"c", "mu"};
      for (unsigned int ig = 0; ig < n_grains; ++ig)
        {
          names.push_back("eta" + std::to_string(ig));
        }
      data_out.add_data_vector(solution, names);

      sintering_operator.add_data_vectors(data_out, solution);

      data_out.build_patches(mapping, this->fe.tensor_degree());

      static unsigned int counter = 0;

      pcout << "Outputing at t = " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };
  };
} // namespace Sintering
