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

#pragma once

#include <deal.II/dofs/dof_handler.h>

namespace hpsint
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim,
            typename VectorType,
            typename MatrixType,
            typename Residual,
            typename NonLinearOperator>
  void
  calc_numeric_tangent(const DoFHandler<dim>   &dof_handler,
                       const NonLinearOperator &nonlinear_operator,
                       const VectorType        &linearization_point,
                       Residual                 nl_residual,
                       MatrixType              &tangent_numeric,
                       const double             epsilon   = 1e-7,
                       const double             tolerance = 1e-12)
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
} // namespace hpsint
