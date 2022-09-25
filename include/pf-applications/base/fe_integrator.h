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
} // namespace dealii
