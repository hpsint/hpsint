#include <deal.II/base/conditional_ostream.h>

#include <iostream>

#include "sintering_model.h"

namespace Test
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename NonLinearOperator>
  void
  calc_numeric_tangent(
    const DoFHandler<dim> &                               dof_handler,
    const NonLinearOperator &                             nonlinear_operator,
    const VectorType &                                    linearization_point,
    std::function<void(const VectorType &, VectorType &)> nl_residual,
    MatrixType &                                          tangent_numeric,
    const double                                          epsilon   = 1e-7,
    const double                                          tolerance = 1e-12)
  {
    VectorType residual;

    nonlinear_operator.initialize_dof_vector(residual);
    nl_residual(linearization_point, residual);

    const VectorType residual0(residual);
    VectorType       state(linearization_point);

    const auto locally_owned_dofs = dof_handler.locally_owned_dofs();
    const auto n_blocks           = state.n_blocks();

    for (unsigned int b = 0; b < n_blocks; ++b)
      for (unsigned int i = 0; i < state.block(b).size(); ++i)
        {
          VectorType residual1(residual);
          residual1 = 0;

          if (locally_owned_dofs.is_element(i))
            state.block(b)[i] += epsilon;

          nl_residual(state, residual1);

          if (locally_owned_dofs.is_element(i))
            state.block(b)[i] -= epsilon;

          for (unsigned int b_ = 0; b_ < n_blocks; ++b_)
            for (unsigned int i_ = 0; i_ < state.block(b).size(); ++i_)
              if (locally_owned_dofs.is_element(i_))
                {
                  if (nonlinear_operator.get_sparsity_pattern().exists(
                        b_ + i_ * n_blocks, b + i * n_blocks))
                    {
                      const auto value =
                        (residual1.block(b_)[i_] - residual0.block(b_)[i_]) /
                        epsilon;

                      if (std::abs(value) > tolerance)
                        tangent_numeric.set(b_ + i_ * n_blocks,
                                            b + i * n_blocks,
                                            value);

                      else if ((b == b_) && (i == i_))
                        tangent_numeric.set(b_ + i_ * n_blocks,
                                            b + i * n_blocks,
                                            1.0);
                    }
                }
        }
  }

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename NonLinearOperator>
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
                   NonLinearOperator>
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
    const std::function<void(const VectorType &, VectorType &)> nl_residual =
      [&](const auto &src, auto &dst) {
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

    if (enable_rbm)
      advection_operator.evaluate_forces_der(solution);

    // At first compute analytic matrix
    FullMatrix<Number> tangent_analytic(n_dofs, n_dofs);
    nonlinear_operator.initialize_system_matrix(false);
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