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

#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/solvers_nonlinear.h>

using namespace dealii;

template <typename VectorType>
void
test()
{
  typename NonLinearSolvers::SNESSolver<VectorType>::AdditionalData
    additional_data;

  NonLinearSolvers::SNESSolver<VectorType> solver(additional_data);

  solver.residual = [&](const VectorType &X, VectorType &F) -> int {
    (void)X;
    (void)F;

    return 0;
  };

  solver.setup_jacobian = [&](const VectorType &X) -> int {
    (void)X;

    return 0;
  };

  solver.solve_with_jacobian =
    [&](const VectorType &src, VectorType &dst, const double) -> int {
    (void)src;
    (void)dst;

    return 0;
  };

  VectorType solution;
  solver.solve(solution);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (false)
    {
      test<LinearAlgebra::distributed::Vector<double>>();
      test<LinearAlgebra::distributed::DynamicBlockVector<double>>();
    }
}
