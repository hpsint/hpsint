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

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>


using namespace dealii;

int
main()
{
  constexpr int dim          = 2;
  constexpr int fe_degree    = 1;
  constexpr int n_components = 4;

  using Number    = double;
  using ValueType = Tensor<1, n_components, Number>;

  QGauss<dim> quad(fe_degree + 1);
  FE_Q<dim>   fe(fe_degree);

  internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(quad, fe);

  const unsigned int n_q_points = shape_info.n_q_points;

  const auto &shape_values = shape_info.data[0].shape_gradients_collocation_eo;

  AlignedVector<ValueType> values(n_q_points);
  AlignedVector<ValueType> gradients(n_q_points * dim);

  internal::EvaluatorTensorProduct<internal::EvaluatorVariant::evaluate_general,
                                   dim,
                                   fe_degree + 1,
                                   fe_degree + 1,
                                   ValueType,
                                   Number>
    phi;

  phi.template apply<0, false, false>(shape_values.data(),
                                      values.data(),
                                      gradients.data() + 0 * n_q_points);
  if constexpr (dim >= 2)
    phi.template apply<1, false, false>(shape_values.data(),
                                        values.data(),
                                        gradients.data() + 1 * n_q_points);
  if constexpr (dim >= 3)
    phi.template apply<2, false, false>(shape_values.data(),
                                        values.data(),
                                        gradients.data() + 2 * n_q_points);

  Tensor<2, dim, Number> jacobian;

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      Tensor<1, n_components, Tensor<1, dim, Number>> gradient;

      for (unsigned int c = 0; c < n_components; ++c)
        for (unsigned int d = 0; d < dim; ++d)
          gradient[c][d] = jacobian[d][d] * gradients[q + d * n_q_points][c];
    }
}
