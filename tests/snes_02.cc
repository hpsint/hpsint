//-----------------------------------------------------------
//
//    Copyright (C) 2022 by the deal.II authors
//
//    This file is part of the deal.II library.
//
//    The deal.II library is free software; you can use it, redistribute
//    it, and/or modify it under the terms of the GNU Lesser General
//    Public License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//    The full text of the license can be found in the file LICENSE.md at
//    the top level directory of deal.II.
//
//-----------------------------------------------------------
//
// Author: Stefano Zampini, King Abdullah University of Science and Technology.

#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_snes.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <cmath>

/**
 * Solves the nonlinear system of equations
 *
 * (x - y^3 + 1)^3 - y^3 = 0
 * x + 2y - 3 = 0
 *
 * using the PETScWrappers::NonlinearSolver class
 * that interfaces PETSc SNES solver object.
 *
 * This code tests the Block interface.
 * See petsc_snes_00.cc for additional information on the callbacks
 */

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  {
    std::cout << "# Test user interface" << std::endl;

    using VectorType = PETScWrappers::MPI::BlockVector;
    using MatrixType = PETScWrappers::MPI::BlockSparseMatrix;
    using Solver     = PETScWrappers::NonlinearSolver<VectorType, MatrixType>;

    Solver solver;

    solver.residual = [&](const VectorType &X, VectorType &F) -> int {
      auto x = X.block(0)[0];
      auto y = X.block(1)[0];

      F.block(0)[0] = std::pow(x - std::pow(y, 3) + 1, 3) - std::pow(y, 3);
      F.block(1)[0] = x + 2 * y - 3;
      F.compress(VectorOperation::insert);
      return 0;
    };

    FullMatrix<double> Jinv(2, 2);

    solver.setup_jacobian = [&](const VectorType &X) -> int {
      auto x    = X.block(0)[0];
      auto y    = X.block(1)[0];
      auto f0_x = 3 * std::pow(x - std::pow(y, 3) + 1, 2);
      auto f0_y = -9 * std::pow(x - std::pow(y, 3) + 1, 2) * std::pow(y, 2) -
                  3 * std::pow(y, 2);
      FullMatrix<double> J(2, 2);
      J(0, 0) = f0_x;
      J(0, 1) = f0_y;
      J(1, 0) = 1;
      J(1, 1) = 2;
      Jinv.invert(J);
      return 0;
    };

    solver.solve_for_jacobian_system = [&](const VectorType &src,
                                           VectorType &      dst) -> int {
      dst.block(0)[0] =
        Jinv(0, 0) * src.block(0)[0] + Jinv(0, 1) * src.block(1)[0];
      dst.block(1)[0] =
        Jinv(1, 0) * src.block(0)[0] + Jinv(1, 1) * src.block(1)[0];
      dst.compress(VectorOperation::insert);
      return 0;
    };

    VectorType x(2, MPI_COMM_SELF, 1, 1);
    x = 0.0;

    auto nit = solver.solve(x);

    std::cout << "#   Solution " << x[0] << ", " << x[1] << std::endl;
    std::cout << "#   Iterations " << nit << std::endl;
  }

  {
    std::cout << "# Test user interface" << std::endl;

    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<double>;
    using Solver     = NonLinearSolvers::SNESSolver<VectorType>;

    typename Solver::AdditionalData additional_data;
    additional_data.max_iter = 0;
    additional_data.abs_tol  = 0;
    additional_data.rel_tol  = 0;

    Solver solver(additional_data);

    solver.residual = [&](const VectorType &X, VectorType &F) -> int {
      auto x = X.block(0)[0];
      auto y = X.block(1)[0];

      F.block(0)[0] = std::pow(x - std::pow(y, 3) + 1, 3) - std::pow(y, 3);
      F.block(1)[0] = x + 2 * y - 3;
      F.compress(VectorOperation::insert);
      return 0;
    };

    FullMatrix<double> Jinv(2, 2);

    solver.setup_jacobian = [&](const VectorType &X) -> int {
      auto x    = X.block(0)[0];
      auto y    = X.block(1)[0];
      auto f0_x = 3 * std::pow(x - std::pow(y, 3) + 1, 2);
      auto f0_y = -9 * std::pow(x - std::pow(y, 3) + 1, 2) * std::pow(y, 2) -
                  3 * std::pow(y, 2);
      FullMatrix<double> J(2, 2);
      J(0, 0) = f0_x;
      J(0, 1) = f0_y;
      J(1, 0) = 1;
      J(1, 1) = 2;
      Jinv.invert(J);
      return 0;
    };

    solver.solve_with_jacobian =
      [&](const VectorType &src, VectorType &dst, const double) -> int {
      dst.block(0)[0] =
        Jinv(0, 0) * src.block(0)[0] + Jinv(0, 1) * src.block(1)[0];
      dst.block(1)[0] =
        Jinv(1, 0) * src.block(0)[0] + Jinv(1, 1) * src.block(1)[0];
      dst.compress(VectorOperation::insert);
      return 0;
    };

    VectorType x(2);
    x.block(0).reinit(1);
    x.block(1).reinit(1);

    x = 0.0;

    auto nit = solver.solve(x);

    std::cout << "#   Solution " << x.block(0)[0] << ", " << x.block(1)[0]
              << std::endl;
    std::cout << "#   Iterations " << nit << std::endl;
  }

  return 0;
}
