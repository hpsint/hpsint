#pragma once

#include <pf-applications/sintering/operator_sintering_base.h>

#include <pf-applications/structural/stvenantkirchhoff.h>
#include <pf-applications/structural/tools.h>

namespace Sintering
{
  using namespace dealii;
  using namespace Structural;

  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperatorCoupled
    : public SinteringOperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        SinteringOperatorCoupled<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = SinteringOperatorCoupled<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorCoupled(
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &   history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const bool                                                  matrix_based)
      : SinteringOperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          SinteringOperatorCoupled<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          data,
          history,
          advection,
          matrix_based)
      , material(1, 0.3, TWO_DIM_TYPE::PLAIN_STRESS)
    {}

    ~SinteringOperatorCoupled()
    {}

    template <bool with_time_derivative = true>
    void
    evaluate_nonlinear_residual(BlockVectorType &      dst,
                                const BlockVectorType &src) const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

#define OPERATION(c, d)                                           \
  MyMatrixFreeTools::cell_loop_wrapper(                           \
    this->matrix_free,                                            \
    &SinteringOperatorCoupled::                                   \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    unsigned int
    n_components() const override
    {
      return this->data.n_components() + dim;
    }

    unsigned int
    n_grains() const
    {
      return this->data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2 + dim;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      AssertDimension(n_comp - 2 - dim, n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &L           = mobility.Lgb();

      // Reinit advection data for the current cells batch
      if (this->advection.enabled())
        this->advection.reinit(cell,
                               static_cast<unsigned int>(n_grains),
                               phi.get_matrix_free());

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          Tensor<1, n_comp, VectorizedArrayType> value_result;
          Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          const auto  value        = phi.get_value(q);
          const auto  gradient     = phi.get_gradient(q);
          const auto &value_lin    = nonlinear_values[cell][q];
          const auto &gradient_lin = nonlinear_gradients[cell][q];

          const auto &c       = value_lin[0];
          const auto &c_grad  = gradient_lin[0];
          const auto &mu_grad = gradient_lin[1];

          const VectorizedArrayType *                etas      = &value_lin[2];
          const Tensor<1, dim, VectorizedArrayType> *etas_grad = nullptr;

          if (SinteringOperatorData<dim, VectorizedArrayType>::
                use_tensorial_mobility ||
              this->advection.enabled())
            etas_grad = &gradient_lin[2];

          const auto etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);

          value_result[0] = value[0] * weight;
          value_result[1] = -value[1] + free_energy.d2f_dc2(c, etas) * value[0];

          gradient_result[0] =
            mobility.M(c, etas, n_grains, c_grad, etas_grad) * gradient[1] +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad * value[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * gradient[0];

          gradient_result[1] = kappa_c * gradient[0];

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[1] +=
                free_energy.d2f_dcdetai(c, etas, ig) * value[ig + 2];

              value_result[ig + 2] +=
                value[ig + 2] * weight +
                L * free_energy.d2f_dcdetai(c, etas, ig) * value[0] +
                L * free_energy.d2f_detai2(c, etas, etaPower2Sum, ig) *
                  value[ig + 2];

              gradient_result[0] +=
                mobility.dM_detai(c, etas, n_grains, c_grad, etas_grad, ig) *
                mu_grad * value[ig + 2];

              gradient_result[ig + 2] = L * kappa_p * gradient[ig + 2];

              for (unsigned int jg = 0; jg < ig; ++jg)
                {
                  const auto d2f_detaidetaj =
                    free_energy.d2f_detaidetaj(c, etas, ig, jg);

                  value_result[ig + 2] += L * d2f_detaidetaj * value[jg + 2];
                  value_result[jg + 2] += L * d2f_detaidetaj * value[ig + 2];
                }

              if (this->advection.enabled() && this->advection.has_velocity(ig))
                {
                  const auto &velocity =
                    this->advection.get_velocity(ig, phi.quadrature_point(q));
                  const auto &velocity_derivative =
                    this->advection.get_velocity_derivative(
                      ig, phi.quadrature_point(q));

                  value_result[0] +=
                    velocity * gradient[0] + velocity_derivative * c_grad;

                  value_result[ig + 2] += velocity * gradient[ig + 2] +
                                          velocity_derivative * etas_grad[ig];
                }
            }

          // Elasticity
          Tensor<2, dim, VectorizedArrayType> H;
          for (unsigned int d = 0; d < dim; d++)
            H[d] = gradient[n_grains + 2 + d];

          const auto E = apply_l(H);

          // TODO: enable this later, this is to check that the matrix is
          // assembled properly.
          // const auto C = c * material.get_dSdE();
          const auto C = material.get_dSdE();

          const auto S = apply_l_transposed<dim>(C * E);

          for (unsigned int d = 0; d < dim; d++)
            gradient_result[n_grains + 2 + d] = S[d];

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

  private:
    template <int n_comp, int n_grains, bool with_time_derivative>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      AssertDimension(n_comp - 2 - dim, n_grains);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      auto time_phi = this->time_integrator.create_cell_intergator(phi);

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto &order       = this->data.time_data.get_order();
      const auto &L           = mobility.Lgb();

      const auto old_solutions = this->history.get_old_solutions();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          if (with_time_derivative)
            for (unsigned int i = 0; i < order; ++i)
              {
                time_phi[i].reinit(cell);
                time_phi[i].read_dof_values_plain(*old_solutions[i]);
                time_phi[i].evaluate(EvaluationFlags::EvaluationFlags::values);
              }

          // Reinit advection data for the current cells batch
          if (this->advection.enabled())
            this->advection.reinit(cell,
                                   static_cast<unsigned int>(n_grains),
                                   matrix_free);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto val  = phi.get_value(q);
              const auto grad = phi.get_gradient(q);

              auto &c      = val[0];
              auto &mu     = val[1];
              auto &c_grad = grad[0];

              std::array<VectorizedArrayType, n_grains> etas;
              std::array<Tensor<1, dim, VectorizedArrayType>, n_grains>
                etas_grad;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]      = val[2 + ig];
                  etas_grad[ig] = grad[2 + ig];
                }

              Tensor<1, n_comp, VectorizedArrayType> value_result;
              Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                gradient_result;

              if (with_time_derivative)
                this->time_integrator.compute_time_derivative(
                  value_result[0], val, time_phi, 0, q);

              value_result[1] = -mu + free_energy.df_dc(c, etas);
              gradient_result[0] =
                mobility.M(c, etas, n_grains, c_grad, etas_grad) * grad[1];
              gradient_result[1] = kappa_c * grad[0];

              // AC equations
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[2 + ig] = L * free_energy.df_detai(c, etas, ig);

                  if (with_time_derivative)
                    this->time_integrator.compute_time_derivative(
                      value_result[2 + ig], val, time_phi, 2 + ig, q);

                  gradient_result[2 + ig] = L * kappa_p * grad[2 + ig];

                  if (this->advection.enabled() &&
                      this->advection.has_velocity(ig))
                    {
                      const auto &velocity =
                        this->advection.get_velocity(ig,
                                                     phi.quadrature_point(q));

                      value_result[0] += velocity * c_grad;
                      value_result[2 + ig] += velocity * grad[2 + ig];
                    }
                }

              // Elasticity
              Tensor<2, dim, VectorizedArrayType> H;
              for (unsigned int d = 0; d < dim; d++)
                H[d] = grad[n_grains + 2 + d];

              const auto E = apply_l(H);

              // update material
              material.reinit(E);
              const auto C = c * material.get_dSdE();

              const auto S = apply_l_transposed<dim>(C * E);

              for (unsigned int d = 0; d < dim; d++)
                gradient_result[n_grains + 2 + d] = S[d];

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    const StVenantKirchhoff<dim, Number, VectorizedArrayType> material;
  };
} // namespace Sintering
