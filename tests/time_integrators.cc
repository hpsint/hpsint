// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <pf-applications/base/debug.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/residual_wrapper.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_linear_tools.h>
#include <pf-applications/lac/solvers_nonlinear.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

#include <pf-applications/time_integration/time_integrator_data.h>
#include <pf-applications/time_integration/time_marching.h>

using namespace dealii;
using namespace TimeIntegration;

using Number          = double;
using BlockVectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;
using VectorType      = LinearAlgebra::distributed::Vector<Number>;

class SinOperator
{
public:
  using value_type = Number;

  struct OpData
  {
    TimeIntegratorData<Number> time_data;
  };

  SinOperator(const OpData &data)
    : data(data)
    , system_matrix(2, 2, 2)
  {
    system_matrix.set(0, 0, 0);
    system_matrix.set(1, 1, 0);
  }

  template <int with_time_derivative>
  void
  evaluate_nonlinear_residual(
    BlockVectorType       &dst,
    const BlockVectorType &src,
    const std::function<void(const unsigned int, const unsigned int)>
      pre_operation = {},
    const std::function<void(const unsigned int, const unsigned int)>
      post_operation = {}) const
  {
    (void)pre_operation;
    (void)post_operation;

    // Index-1 DAE
    // dx/dt = -x + y;
    // y = sin(t)

    const auto &weight_alg   = data.time_data.get_algebraic_weight();
    const auto &current_time = data.time_data.get_current_time();

    // There should be a proper time derivative evaluation here to test implicit
    // schemes too

    dst[0] = src[0] - std::sin(current_time);
    dst[1] = weight_alg * src[1] - std::sin(current_time);
  }

  TrilinosWrappers::SparseMatrix &
  get_system_matrix()
  {
    return system_matrix;
  }

  const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const
  {
    return system_matrix;
  }

  void
  initialize_dof_vector(BlockVectorType &vector) const
  {
    vector.reinit(n_components());
    for (unsigned int b = 0; b < vector.n_blocks(); ++b)
      vector.block(b).reinit(1);
    vector.collect_sizes();
  }

  const OpData &
  get_data() const
  {
    return data;
  }

  unsigned int
  n_components() const
  {
    return 2;
  }

  EquationType
  equation_type(const unsigned int component) const
  {
    return component == 0 ? EquationType::TimeDependent :
                            EquationType::Stationary;
  }

private:
  const OpData                  &data;
  TrilinosWrappers::SparseMatrix system_matrix;
};

struct MarchingSet
{
  std::string     label;
  BlockVectorType solution;
  // Statistics is made shared to make it heap stable, such that the reference
  // to it inside the marching object did not get broken if the MarchingSet
  // object has been moved
  std::shared_ptr<NonLinearSolvers::NewtonSolverSolverControl> statistics;
  std::unique_ptr<TimeMarching<BlockVectorType>>               marching;

  template <typename... Args>
  MarchingSet(const std::string                     &label_in,
              const BlockVectorType                 &solution_in,
              const NonLinearSolvers::NonLinearData &nonlinear_data,
              MyTimerOutput                         &timer,
              std::unique_ptr<ExplicitScheme>        scheme,
              Args &&...args)
    : label(label_in)
    , solution(solution_in)
    , statistics(std::make_shared<NonLinearSolvers::NewtonSolverSolverControl>(
        nonlinear_data.nl_max_iter,
        nonlinear_data.nl_abs_tol,
        nonlinear_data.nl_rel_tol))
    , marching(
        std::make_unique<
          TimeMarchingExplicit<BlockVectorType, SinOperator, std::ostream>>(
          std::forward<Args>(args)...,
          std::move(scheme),
          *statistics,
          timer,
          std::cout))
  {}
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  // Dummy dof handler
  DoFHandler<1>             dof_handler;
  AffineConstraints<Number> constraints;
  constraints.close();

  SinOperator::OpData data;

  // Set time settings
  const Number dt = 1e-3;
  data.time_data.update_dt(dt);

  SinOperator nonlinear_operator(data);

  const NonLinearSolvers::NonLinearData nonlinear_data;

  NonLinearSolvers::ResidualWrapperDirect<Number, SinOperator, false>
    residual_wrapper(nonlinear_operator);

  MyTimerOutput timer(std::cout);

  // Time schemes to test
  std::map<std::string, std::unique_ptr<ExplicitScheme>> schemes;
  schemes.emplace("FE", std::make_unique<ForwardEulerScheme>());
  schemes.emplace("RK4", std::make_unique<RungeKutta4Scheme>());

  // Create time marching for each scheme with the corresponding state vector
  std::vector<MarchingSet> time_marchings;

  // Initialize solution vector - we will copy it
  BlockVectorType solution(2);
  nonlinear_operator.initialize_dof_vector(solution);

  // Output
  TableHandler table;
  table.add_value("Step", 0);
  table.add_value("Time", 0);
  table.add_value("Exact x", 0);
  table.add_value("Exact y", 0);
  table.set_scientific("Exact x", true);
  table.set_scientific("Exact y", true);

  for (auto &[label, scheme] : schemes)
    {
      time_marchings.emplace_back(label,
                                  solution,
                                  nonlinear_data,
                                  timer,
                                  std::move(scheme),
                                  nonlinear_operator,
                                  residual_wrapper,
                                  constraints);

      table.add_value(label + " x", 0);
      table.add_value(label + " y", 0);
      table.set_scientific(label + " x", true);
      table.set_scientific(label + " y", true);
      table.add_value(label + " n_evals", 0);
    }

  auto exact_solution = [](Number t) {
    return std::vector{0.5 * std::exp(-t) + 0.5 * (std::sin(t) - std::cos(t)),
                       std::sin(t)};
  };

  const unsigned int n_steps = 4;
  Number             tc      = 0;
  for (unsigned int step = 1; step <= n_steps; ++step, tc += dt)
    {
      const auto tn = tc + dt;

      table.add_value("Step", step);
      table.add_value("Time", tn);

      const auto exact_sol = exact_solution(tn);
      table.add_value("Exact x", exact_sol[0]);
      table.add_value("Exact y", exact_sol[1]);

      for (auto &marching_set : time_marchings)
        {
          // Reset time data, since it is modified inside the time marching
          data.time_data.set_current_time(tc);

          // Make time step
          marching_set.marching->make_step(marching_set.solution);

          table.add_value(marching_set.label + " x", marching_set.solution[0]);
          table.add_value(marching_set.label + " y", marching_set.solution[1]);
          table.add_value(marching_set.label + " n_evals",
                          marching_set.statistics->n_residual_evaluations());
        }
    }

  table.write_text(std::cout,
                   TableHandler::TextOutputFormat::table_with_headers);

  return 0;
}