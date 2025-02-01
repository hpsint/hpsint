// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#define MAX_SINTERING_GRAINS 2
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1
#define WITH_TENSORIAL_MOBILITY

#include <deal.II/base/mpi.h>

#include <pf-applications/sintering/operator_grand_potential_greenquist.h>

#include <pf-applications/tests/tangent_tester.h>

using namespace dealii;
using namespace Sintering;

using Number              = double;
using VectorizedArrayType = VectorizedArray<Number>;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim = 2;

  using NonLinearOperator =
    GreenquistGrandPotentialOperator<dim, Number, VectorizedArrayType>;

  using OperatorFreeEnergy = GreenquistFreeEnergy<VectorizedArrayType>;

  // TODO: Understand why the tolerance is much worse in debug on github
#ifndef NDEBUG
  const double tol_abs_no_rbm = 1e2;
  const double tol_rel_no_rbm = 1e-4;
#else
  const double tol_abs_no_rbm = 1e0;
  const double tol_rel_no_rbm = 1e-5;
#endif

  Test::check_tangent<dim,
                      Number,
                      VectorizedArrayType,
                      NonLinearOperator,
                      OperatorFreeEnergy>(false,
                                          "WITHOUT RBM",
                                          tol_abs_no_rbm,
                                          tol_rel_no_rbm);
}