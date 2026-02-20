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

#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class PostprocOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          PostprocOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = PostprocOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    PostprocOperator(
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const AffineConstraints<Number>                       &constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const bool                                             matrix_based)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     PostprocOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "postproc_op",
          matrix_based)
      , data(data)
    {}

    ~PostprocOperator()
    {}

    void
    evaluate_rhs(BlockVectorType &dst, const BlockVectorType &src) const
    {
      MyScope scope(this->timer, "postproc_op::residual", this->do_timing);

#define OPERATION(c, d)                       \
  MyMatrixFreeTools::cell_loop_wrapper(       \
    this->matrix_free,                        \
    &PostprocOperator::do_evaluate_rhs<c, d>, \
    this,                                     \
    dst,                                      \
    src,                                      \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    void
    do_update() override
    {
      if (this->matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    unsigned int
    n_components() const override
    {
      return 4;
    }

    unsigned int
    n_grains() const
    {
      return data.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 4;
    }

    virtual EquationType
    equation_type(const unsigned int component) const override
    {
      (void)component;
      return EquationType::Stationary;
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim>                &data_out,
                               const BlockVectorType       &vec,
                               const std::set<std::string> &fields_list) const
    {
      (void)fields_list;

      std::vector<std::string> names = {"div_vol",
                                        "div_vap",
                                        "div_surf",
                                        "div_gb"};

      AssertDimension(names.size(), vec.n_blocks());

      for (unsigned int i = 0; i < names.size(); ++i)
        data_out.add_data_vector(this->matrix_free.get_dof_handler(
                                   this->dof_index),
                                 vec.block(i),
                                 names[i]);
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_rhs(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free, this->dof_index);
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi_post(
        matrix_free, this->dof_index);

      const auto &mobility = this->data.get_mobility();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_post.reinit(cell);

          phi_sint.gather_evaluate(
            src,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi_post.n_q_points; ++q)
            {
              const auto val  = phi_sint.get_value(q);
              const auto grad = phi_sint.get_gradient(q);

              auto &c       = val[0];
              auto &c_grad  = grad[0];
              auto &mu_grad = grad[1];

              std::array<VectorizedArrayType, n_grains> etas;
              std::array<Tensor<1, dim, VectorizedArrayType>, n_grains>
                etas_grad;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]      = val[2 + ig];
                  etas_grad[ig] = grad[2 + ig];
                }

              Tensor<1, 4, Tensor<1, dim, VectorizedArrayType>> gradient_result;

              gradient_result[0] = -mobility.M_vol(c) * mu_grad;
              gradient_result[1] = -mobility.M_vap(c) * mu_grad;
              gradient_result[2] = -mobility.M_surf(c, c_grad) * mu_grad;
              gradient_result[3] =
                -mobility.M_gb(etas, n_grains, etas_grad) * mu_grad;

              phi_post.submit_gradient(gradient_result, q);
            }
          phi_post.integrate_scatter(
            EvaluationFlags::EvaluationFlags::gradients, dst);
        }
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
  };

} // namespace Sintering
