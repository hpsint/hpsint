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

      : f0(make_vectorized_array(dim == 3 ?
                                   E * (1 - nu) / (1 + nu) / (1 - 2 * nu) :
                                   (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                                      E * (1) / (1 - nu * nu) :
                                      E * (1 - nu) / (1 + nu) / (1 - 2 * nu))))
      , f1(make_vectorized_array(dim == 3 ?
                                   E * (nu) / (1 + nu) / (1 - 2 * nu) :
                                   (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                                      E * (nu) / (1 - nu * nu) :
                                      E * (nu) / (1 + nu) / (1 - 2 * nu))))
      , f2(make_vectorized_array(
          dim == 3 ? E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu) :
                     (two_dim_type == TWO_DIM_TYPE::PLAIN_STRESS ?
                        E * (1 - nu) / 2 / (1 - nu * nu) :
                        E * (1 - 2 * nu) / 2 / (1 + nu) / (1 - 2 * nu))))
    {
      for (unsigned int i = 0; i < dim; i++)
        for (unsigned int j = 0; j < dim; j++)
          if (i == j)
            C[i][j] = f0;
          else
            C[i][j] = f1;

      for (unsigned int i = dim; i < voigt_size<dim>; i++)
        C[i][i] = f2;
    }

    void
    reinit(const Tensor<1, voigt_size<dim>, VectorizedArrayType> &E_in)
      const override
    {
      E = E_in;
    }

    Tensor<1, voigt_size<dim>, VectorizedArrayType>
    get_S() const override
    {
      return C * E;
    }

    const Tensor<2, voigt_size<dim>, VectorizedArrayType>
    get_dSdE() const override
    {
      return C;
    }

  private:
    const VectorizedArrayType f0;
    const VectorizedArrayType f1;
    const VectorizedArrayType f2;

    mutable Tensor<2, voigt_size<dim>, VectorizedArrayType> C;
    mutable Tensor<1, voigt_size<dim>, VectorizedArrayType> E;
  };
} // namespace Structural