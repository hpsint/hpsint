#pragma once

#include <pf-applications/structural/material.h>

namespace Structural
{
  template <int dim, typename Number, typename VectorizedArrayType>
  class StVenantKirchhoff : public Material<dim, VectorizedArrayType>
  {
  public:
    StVenantKirchhoff(Number       E,
                      Number       nu,
                      TWO_DIM_TYPE two_dim_type = TWO_DIM_TYPE::NONE)

      : two_dim_type(two_dim_type)
      , lambda(E * nu / (1 + nu) / (1 - (dim - 1) * nu))
      , mu_x_2(E / (1 + nu))
    {
      const VectorizedArrayType f0 =
        dim == 3 ? E * (1 - nu) / (1 + nu) / (1 - 2 * nu) :
                   (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                      E * (1) / (1 - nu * nu) :
                      E * (1 - nu) / (1 + nu) / (1 - 2 * nu));
      const VectorizedArrayType f1 =
        dim == 3 ? E * (nu) / (1 + nu) / (1 - 2 * nu) :
                   (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                      E * (nu) / (1 - nu * nu) :
                      E * (nu) / (1 + nu) / (1 - 2 * nu));
      const VectorizedArrayType f2 =
        dim == 3 ? E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu) :
                   (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                      E * (1 - nu) / 2 / (1 - nu * nu) :
                      E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu));

      for (unsigned int i = 0; i < dim; i++)
        for (unsigned int j = 0; j < dim; j++)
          if (i == j)
            C[i][j] = f0;
          else
            C[i][j] = f1;

      for (unsigned int i = dim; i < voigt_size<dim>; i++)
        C[i][i] = f2;
    }

    Tensor<2, dim, VectorizedArrayType>
    get_S(const Tensor<2, dim, VectorizedArrayType> &E) const override
    {
      if (dim == 3 || two_dim_type == TWO_DIM_TYPE::NONE)
        {
          Tensor<2, dim, VectorizedArrayType> stress;

          VectorizedArrayType trace = E[0][0];

          for (unsigned int i = 1; i < dim; ++i)
            trace += E[i][i];

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              stress[i][j] = mu_x_2 * E[i][j];

          for (unsigned int i = 0; i < dim; ++i)
            stress[i][i] += lambda * trace;

          return stress;
        }
      else
        {
          return Structural::apply_l_transposed<dim>(C *
                                                     Structural::apply_l(E));
        }
    }

  private:
    const TWO_DIM_TYPE                              two_dim_type;
    const VectorizedArrayType                       lambda;
    const VectorizedArrayType                       mu_x_2;
    Tensor<2, voigt_size<dim>, VectorizedArrayType> C;
  };
} // namespace Structural
