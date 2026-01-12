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

#pragma once

#include <deal.II/lac/solver_control.h>

#include <pf-applications/lac/preconditioners.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

namespace LinearSolvers
{
  using namespace dealii;
  using namespace NonLinearSolvers;

  template <typename Number>
  struct LinearSolverWrapper
  {
    std::unique_ptr<ReductionControl>                            solver_control;
    std::unique_ptr<Preconditioners::PreconditionerBase<Number>> preconditioner;
    std::unique_ptr<LinearSolvers::LinearSolverBase<Number>>     linear_solver;
  };

  template <typename Operator,
            template <typename Operator_>
            typename Preconditioner,
            template <typename Operator_>
            typename Solver>
  LinearSolverWrapper<typename Operator::value_type>
  wrap_solver_with_preconditioner(const Operator      &op,
                                  const NonLinearData &nonlinear_params)
  {
    LinearSolverWrapper<typename Operator::value_type> wrapper;

    wrapper.solver_control =
      std::make_unique<ReductionControl>(nonlinear_params.l_max_iter,
                                         nonlinear_params.l_abs_tol,
                                         nonlinear_params.l_rel_tol);

    // Identity, InverseDiagonalMatrix, IC
    wrapper.preconditioner = std::make_unique<Preconditioner<Operator>>(op);

    wrapper.linear_solver =
      std::make_unique<Solver<Operator>>(op,
                                         *wrapper.preconditioner,
                                         *wrapper.solver_control);

    return wrapper;
  }
} // namespace LinearSolvers
