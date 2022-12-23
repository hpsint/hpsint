// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

// Test performance of the sintering operator

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

//#define USE_FE_Q_iso_Q1

#ifdef USE_FE_Q_iso_Q1
#  define FE_DEGREE 2
#  define N_Q_POINTS_1D FE_DEGREE * 2
#else
#  define FE_DEGREE 1
#  define N_Q_POINTS_1D FE_DEGREE + 1
#endif

#define WITH_TIMING
//#define WITH_TIMING_OUTPUT

#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <pf-applications/base/revision.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/mobility.h>
#include <pf-applications/sintering/operator_sintering_generic.h>
#include <pf-applications/sintering/sintering_data.h>

using namespace dealii;
using namespace Sintering;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int dim    = SINTERING_DIM;
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  // some arbitrary constants
  const double A                      = 16;
  const double B                      = 1;
  const double kappa_c                = 1;
  const double kappa_p                = 0.5;
  const double Mvol                   = 1e-2;
  const double Mvap                   = 1e-10;
  const double Msurf                  = 4;
  const double Mgb                    = 0.4;
  const double L                      = 1;
  const double time_integration_order = 1;

  AffineConstraints<Number> constraints;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

  const std::shared_ptr<MobilityProvider> mobility_provider =
    std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

  TimeIntegration::SolutionHistory<VectorType> solution_history(
    time_integration_order + 1);

  SinteringOperatorData<dim, VectorizedArrayType> data(
    A, B, kappa_c, kappa_p, mobility_provider, time_integration_order);

  AdvectionMechanism<dim, Number, VectorizedArrayType> advection;

  SinteringOperatorGeneric<dim, Number, VectorizedArrayType> op_sintering(
    matrix_free, constraints, data, solution_history, advection, false);
}
