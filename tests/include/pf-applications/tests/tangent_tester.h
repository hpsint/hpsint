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

#include <deal.II/base/conditional_ostream.h>

#include <pf-applications/lac/evaluation.h>

#include <pf-applications/tests/sintering_model.h>

#include <iostream>

namespace Test
{
  using namespace dealii;
  using namespace hpsint;
  using namespace Sintering;

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename NonLinearOperator,
            typename FreeEnergy>
  void
  check_tangent(const bool        enable_rbm,
                const std::string prefix,
                const double      tol_abs = 1e-3,
                const double      tol_rel = 1e-6)
  {
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    SinteringModel<dim,
                   Number,
                   VectorType,
                   VectorizedArrayType,
                   NonLinearOperator,
                   FreeEnergy>
      sintering_model(enable_rbm);

    auto &nonlinear_operator = sintering_model.get_nonlinear_operator();
    auto &advection_operator = sintering_model.get_advection_operator();
    auto &grain_tracker      = sintering_model.get_grain_tracker();
    auto &dof_handler        = sintering_model.get_dof_handler();
    auto &solution           = sintering_model.get_solution();

    const auto comm = dof_handler.get_communicator();

    const bool is_zero_rank = Utilities::MPI::this_mpi_process(comm) == 0;
    ConditionalOStream pcout(std::cout, is_zero_rank);

    // How to compute residual
    auto nl_residual = [&](const auto &src, auto &dst) {
      if (enable_rbm)
        advection_operator.evaluate_forces(src);

      nonlinear_operator.evaluate_nonlinear_residual(dst, src);
    };

    // Evaluate residual (used for debug and to init advection data structures)
    VectorType residual;
    nonlinear_operator.initialize_dof_vector(residual);
    nl_residual(solution, residual);

    const unsigned int n_blocks = solution.n_blocks();
    const unsigned int n_dofs   = dof_handler.n_dofs() * n_blocks;

    // At first compute analytic matrix
    FullMatrix<Number> tangent_analytic(n_dofs, n_dofs);
    nonlinear_operator.initialize_system_matrix(true, false);
    tangent_analytic.copy_from(nonlinear_operator.get_system_matrix());

    // Then compute analytic matrix
    FullMatrix<Number> tangent_numeric(n_dofs, n_dofs);
    calc_numeric_tangent(
      dof_handler, nonlinear_operator, solution, nl_residual, tangent_numeric);

    auto tangent_diff(tangent_analytic);
    tangent_diff.add(Number(-1.), tangent_numeric);

    const auto nrm_analytic = tangent_analytic.l1_norm();
    const auto error_abs    = tangent_diff.l1_norm();
    const auto error_rel = nrm_analytic > 1e-20 ? error_abs / nrm_analytic : 0;

    const bool is_correct = error_abs < tol_abs && error_rel < tol_rel;
    const bool all_correct =
      (Utilities::MPI::sum<unsigned int>(is_correct, comm) ==
       Utilities::MPI::n_mpi_processes(comm));

    pcout << "Tangent " << prefix << " is " << (all_correct ? "OK" : "ERROR")
          << " with tol_abs = " << tol_abs << " and tol_rel = " << tol_rel
          << (all_correct ? "" :
                            (": " + std::to_string(error_abs) + "/" +
                             std::to_string(error_rel)))
          << std::endl;

    if (!all_correct)
      {
        pcout << std::endl;
        grain_tracker.print_current_grains(pcout, true);
        pcout << std::endl;

        std::ostringstream ss;

        ss << std::endl;
        ss << "===== Output from rank "
           << Utilities::MPI::this_mpi_process(comm)
           << " (total = " << Utilities::MPI::n_mpi_processes(comm)
           << ") =====" << std::endl;

        ss << std::endl;
        ss << "diff L2 norm absolute = " << error_abs
           << " (tol_abs = " << tol_abs << ") - "
           << (error_abs < tol_abs ? "OK" : "ERROR") << std::endl;
        ss << "diff L2 norm relative = " << error_rel
           << " (tol_rel = " << tol_rel << ") - "
           << (error_rel < tol_rel ? "OK" : "ERROR") << std::endl;
        ss << std::endl;

        ss << std::endl << "Tangent analytic:" << std::endl;
        tangent_analytic.print_formatted(ss);

        ss << std::endl << "Tangent numeric:" << std::endl;
        tangent_numeric.print_formatted(ss);

        ss << std::endl << "Tangent diff:" << std::endl;
        tangent_diff.print_formatted(ss);

        ss << std::endl << "Solution:" << std::endl;
        for (unsigned int c = 0; c < solution.n_blocks(); ++c)
          {
            ss << "block " << c << ": ";
            solution.block(c).print(ss);
            ss << std::endl;
          }

        ss << std::endl << "Residual:" << std::endl;
        for (unsigned int c = 0; c < residual.n_blocks(); ++c)
          {
            ss << "block " << c << ": ";
            residual.block(c).print(ss);
            ss << std::endl;
          }

        auto all_prints = Utilities::MPI::gather(comm, ss.str());

        for (const auto &entry : all_prints)
          pcout << entry;
      }
  }
} // namespace Test