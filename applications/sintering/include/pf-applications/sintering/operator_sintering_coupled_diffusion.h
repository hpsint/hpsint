#pragma once

#include <pf-applications/sintering/inelastic.h>
#include <pf-applications/sintering/operator_sintering_coupled_base.h>

#include <pf-applications/structural/tools.h>
namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperatorCoupledDiffusion
    : public SinteringOperatorCoupledBase<
        dim,
        Number,
        VectorizedArrayType,
        SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>>
  {
  public:
    using T =
      SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorCoupledDiffusion(
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &  data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &history,
      const bool                                               matrix_based,
      const double                                             E  = 1.0,
      const double                                             nu = 0.25)
      : SinteringOperatorCoupledBase<
          dim,
          Number,
          VectorizedArrayType,
          SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          data,
          history,
          matrix_based,
          E,
          nu)
      , inelastic(data, /*rho = */ 1.0, /*time_start = */ 0.0)
    {}

    SinteringOperatorCoupledDiffusion() = default;

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
    &SinteringOperatorCoupledDiffusion::                          \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      // Apply manual constraints
      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : this->get_zero_constraints_indices()[d])
          dst.block(this->data.n_components() + n_additional_components() + d)
            .local_element(index) = 0.0;
    }

    unsigned int
    n_additional_components() const override
    {
      return 2;
    }

    unsigned int
    n_components() const override
    {
      return this->data.n_components() + n_additional_components() + dim;
    }

    unsigned int
    n_grains() const
    {
      return this->data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2 + 2 + dim;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      AssertDimension(n_comp - 2 - n_additional_components() - dim, n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &L           = mobility.Lgb();
      const auto  dt          = this->data.time_data.get_current_dt();
      const auto  inv_dt      = 1. / this->data.time_data.get_current_dt();

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

          const auto &div_gb  = value_lin[this->data.n_components() + 0];
          const auto &div_vol = value_lin[this->data.n_components() + 1];

          const VectorizedArrayType *                etas = &value_lin[2];
          const Tensor<1, dim, VectorizedArrayType> *etas_grad =
            &gradient_lin[2];

          const auto etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);

          // Displacement field
          Tensor<1, dim, VectorizedArrayType> v_adv;
          Tensor<1, dim, VectorizedArrayType> v_adv_lin;
          for (unsigned int d = 0; d < dim; ++d)
            {
              v_adv[d]     = value[this->data.n_components() +
                               n_additional_components() + d];
              v_adv_lin[d] = value_lin[this->data.n_components() +
                                       n_additional_components() + d];
            }

          // Advection velocity
          v_adv *= inv_dt;
          v_adv_lin *= inv_dt;

          value_result[0] = value[0] * weight;
          value_result[1] = -value[1] + free_energy.d2f_dc2(c, etas) * value[0];

          gradient_result[0] =
            mobility.M(c, etas, n_grains, c_grad, etas_grad) * gradient[1] +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad * value[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * gradient[0];

          gradient_result[1] = kappa_c * gradient[0];

          // Diffusion fluxes terms
          value_result[this->data.n_components() + 0] =
            value[this->data.n_components() + 0];
          gradient_result[this->data.n_components() + 0] =
            -mobility.M_gb(etas, n_grains, etas_grad) * gradient[1];

          value_result[this->data.n_components() + 1] =
            value[this->data.n_components() + 1];
          gradient_result[this->data.n_components() + 1] =
            -mobility.M_vol(c) * gradient[1] -
            mobility.dM_vol_dc(c) * mu_grad * value[0];

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

              gradient_result[this->data.n_components() + 0] -=
                mobility.dM_detai(c, etas, n_grains, c_grad, etas_grad, ig) *
                mu_grad * value[ig + 2];

              for (unsigned int jg = 0; jg < ig; ++jg)
                {
                  const auto d2f_detaidetaj =
                    free_energy.d2f_detaidetaj(c, etas, ig, jg);

                  value_result[ig + 2] += L * d2f_detaidetaj * value[jg + 2];
                  value_result[jg + 2] += L * d2f_detaidetaj * value[ig + 2];
                }

              if (true)
                {
                  value_result[0] += v_adv * c_grad + v_adv_lin * gradient[0];

                  value_result[ig + 2] +=
                    v_adv * etas_grad[ig] + v_adv_lin * gradient[ig + 2];
                }
            }

          // Elasticity
          Tensor<2, dim, VectorizedArrayType> H;
          for (unsigned int d = 0; d < dim; d++)
            H[d] = gradient[this->data.n_components() +
                            n_additional_components() + d];

          const auto E = Structural::apply_l(H);
          const auto C = this->dSdE(E, c);
          const auto S = Structural::apply_l_transposed<dim>(C * E);

          for (unsigned int d = 0; d < dim; d++)
            gradient_result[this->data.n_components() +
                            n_additional_components() + d] = S[d];

          // Diffusion strain
          auto eps_inelastic_deriv =
            inelastic.flux_eps_dot_dc(
              c, etas, n_grains, etas_grad, div_gb, div_vol) *
              value[0] +
            inelastic.flux_eps_dot_ddiv_gb(c, etas, n_grains, etas_grad) *
              value[this->data.n_components() + 0] +
            inelastic.flux_eps_dot_ddiv_vol(c, etas, n_grains) *
              value[this->data.n_components() + 1];

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            eps_inelastic_deriv +=
              inelastic.flux_eps_dot_detai(
                c, etas, n_grains, etas_grad, div_gb, div_vol, ig) *
              value[ig + 2];

          eps_inelastic_deriv *= dt;

          const auto eps_voigt = Structural::apply_l(eps_inelastic_deriv);
          const auto S_deriv =
            Structural::apply_l_transposed<dim>(C * eps_voigt);

          for (unsigned int d = 0; d < dim; d++)
            gradient_result[this->data.n_components() +
                            n_additional_components() + d] += S_deriv[d];

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
      AssertDimension(n_comp - 2 - n_additional_components() - dim, n_grains);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      auto time_phi = this->time_integrator.create_cell_intergator(phi);

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto &order       = this->data.time_data.get_order();
      const auto &L           = mobility.Lgb();
      const auto  dt          = this->data.time_data.get_current_dt();
      const auto  inv_dt      = 1. / this->data.time_data.get_current_dt();

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

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto val  = phi.get_value(q);
              const auto grad = phi.get_gradient(q);

              auto &c       = val[0];
              auto &mu      = val[1];
              auto &div_gb  = val[this->data.n_components() + 0];
              auto &div_vol = val[this->data.n_components() + 1];
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

              // Displacement field
              Tensor<1, dim, VectorizedArrayType> v_adv;
              for (unsigned int d = 0; d < dim; ++d)
                v_adv[d] = val[this->data.n_components() +
                               n_additional_components() + d];

              // Advection velocity
              v_adv *= inv_dt;

              Tensor<1, n_comp, VectorizedArrayType> value_result;
              Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                gradient_result;

              if (with_time_derivative)
                this->time_integrator.compute_time_derivative(
                  value_result[0], val, time_phi, 0, q);

              value_result[1] = -mu + free_energy.df_dc(c, etas);
              gradient_result[0] =
                mobility.M(c, etas, n_grains, c_grad, etas_grad) * mu_grad;
              gradient_result[1] = kappa_c * c_grad;

              // AC equations
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[2 + ig] = L * free_energy.df_detai(c, etas, ig);

                  if (with_time_derivative)
                    this->time_integrator.compute_time_derivative(
                      value_result[2 + ig], val, time_phi, 2 + ig, q);

                  gradient_result[2 + ig] = L * kappa_p * grad[2 + ig];

                  if (true) // Add advection terms
                    {
                      value_result[0] += v_adv * c_grad;
                      value_result[2 + ig] += v_adv * grad[2 + ig];
                    }
                }

              value_result[this->data.n_components() + 0] = div_gb;
              gradient_result[this->data.n_components() + 0] =
                -mobility.M_gb(etas, n_grains, etas_grad) * mu_grad;

              value_result[this->data.n_components() + 1] = div_vol;
              gradient_result[this->data.n_components() + 1] =
                -mobility.M_vol(c) * mu_grad;

              // Elasticity
              Tensor<2, dim, VectorizedArrayType> H;
              for (unsigned int d = 0; d < dim; d++)
                H[d] = grad[this->data.n_components() +
                            n_additional_components() + d];

              // Diffusion strain
              auto eps_inelastic = inelastic.flux_eps_dot(
                c, etas, n_grains, etas_grad, div_gb, div_vol);

              eps_inelastic *= dt;

              H += eps_inelastic;

              const auto E = Structural::apply_l(H);
              const auto C = this->dSdE(E, c);
              const auto S = Structural::apply_l_transposed<dim>(C * E);

              for (unsigned int d = 0; d < dim; d++)
                gradient_result[this->data.n_components() +
                                n_additional_components() + d] = S[d];

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    InelasticStrains<dim, Number, VectorizedArrayType> inelastic;
  };
} // namespace Sintering
