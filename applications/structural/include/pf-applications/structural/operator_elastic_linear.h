#pragma once

#include <pf-applications/sintering/operator_base.h>

#include <pf-applications/structural/stvenantkirchhoff.h>
#include <pf-applications/structural/tools.h>

namespace Structural
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim, typename Number, typename VectorizedArrayType>
  class LinearElasticOperator
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        LinearElasticOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = LinearElasticOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    using ConstraintsCallback =
      std::function<void(const DoFHandler<dim> &dof_handler,
                         AffineConstraints<Number> &)>;

    using ExternalLoadingCallback =
      std::function<Tensor<1, dim, VectorizedArrayType>(
        FECellIntegrator<dim, dim, Number, VectorizedArrayType> &phi,
        const unsigned int                                       q)>;

    LinearElasticOperator(
      const double                                        E,
      const double                                        nu,
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const bool                                          matrix_based,
      ConstraintsCallback                                 imposition = {},
      ExternalLoadingCallback                             loading    = {})
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     LinearElasticOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "elastic_linear_op",
          matrix_based)
      , material(E, nu, TWO_DIM_TYPE::PLAIN_STRAIN)
      , constraints_imposition(imposition)
      , external_loading(loading)
    {}

    ~LinearElasticOperator()
    {}

    void
    evaluate_nonlinear_residual(BlockVectorType &      dst,
                                const BlockVectorType &src) const
    {
      MyScope scope(this->timer,
                    "elastic_linear_op::nonlinear_residual",
                    this->do_timing);

      dst = 0.0;

      src.update_ghost_values();

      do_evaluate_nonlinear_residual(this->matrix_free,
                                     dst,
                                     src,
                                     std::pair<unsigned int, unsigned int>{
                                       0, this->matrix_free.n_cell_batches()});

      src.zero_out_ghost_values();
      dst.compress(VectorOperation::add);
    }

    void
    do_update()
    {
      if (this->matrix_based)
        this->initialize_system_matrix(); // assemble matrix
    }

    unsigned int
    n_components() const override
    {
      return dim;
    }

    unsigned int
    n_grains() const
    {
      return dim;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      (void)n_grains;
      return dim;
    }

    void
    add_matrix_constraints(
      const DoFHandler<dim> &    dof_handler,
      AffineConstraints<Number> &matrix_constraints) const override
    {
      if (constraints_imposition)
        constraints_imposition(dof_handler, matrix_constraints);
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      Tensor<1, dim, VectorizedArrayType> zero_result;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto grad = phi.get_gradient(q);

          const auto E = apply_l(grad);

          // update material
          material.reinit(E);

          const auto C = material.get_dSdE();
          const auto S = apply_l_transposed<dim>(C * E);

          phi.submit_gradient(S, q);
          phi.submit_value(zero_result, q);
        }
    }

  private:
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, dim, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::values |
                                EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto C = material.get_dSdE();

              const auto grad = phi.get_gradient(q);
              const auto E    = apply_l(grad);
              // update material
              material.reinit(E);

              const auto S = apply_l_transposed<dim>(C * E);

              phi.submit_gradient(S, q);

              // External loading
              if (external_loading)
                {
                  auto applied_force = external_loading(phi, q);
                  phi.submit_value(applied_force, q);
                }
            }
          phi.integrate_scatter(EvaluationFlags::values |
                                  EvaluationFlags::gradients,
                                dst);
        }
    }

    const StVenantKirchhoff<dim, Number, VectorizedArrayType> material;

    const ConstraintsCallback     constraints_imposition;
    const ExternalLoadingCallback external_loading;
  };
} // namespace Structural
