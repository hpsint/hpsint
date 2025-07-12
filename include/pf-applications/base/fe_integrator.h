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

#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>

namespace dealii
{
#if defined(FE_DEGREE) && defined(N_Q_POINTS_1D)
  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  using FECellIntegrator = FEEvaluation<dim,
                                        FE_DEGREE,
                                        N_Q_POINTS_1D,
                                        n_comp,
                                        Number,
                                        VectorizedArrayType>;
#else
  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  using FECellIntegrator =
    FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType>;
#endif

  template <int dim, int n_comp, typename Number, typename VectorizedArrayType>
  using FECellIntegratorValue =
    typename FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>::
      value_type;
} // namespace dealii
