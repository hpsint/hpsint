#pragma once

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/sintering_data.h>

#include <pf-applications/time_integration/solution_history.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          SinteringOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = SinteringOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &   history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const bool                                                  matrix_based)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     SinteringOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "sintering_op",
          matrix_based)
      , data(data)
      , history(history)
      , time_integrator(data.time_data, history)
      , advection(advection)
    {}

    ~SinteringOperator()
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
    &SinteringOperator::                                          \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    void
    do_update()
    {
      if (this->matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &
    get_data() const
    {
      return data;
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim> &               data_out,
                               const BlockVectorType &      vec,
                               const std::set<std::string> &fields_list) const
    {
      AssertDimension(n_comp - 2, n_grains);

      // Possible output options
      enum OutputFields
      {
        FieldBnds,
        FieldDt,
        FieldD2f,
        FieldM,
        FieldDM,
        FieldKappa,
        FieldL,
        FieldF,
        FieldFlux
      };

      constexpr unsigned int n_data_variants = 9;

      const std::array<std::tuple<std::string, OutputFields, unsigned int>,
                       n_data_variants>
        possible_entries = {
          {{"bnds", FieldBnds, 1},
           {"dt", FieldDt, 1},
           {"d2f", FieldD2f, 1 + 2 * n_grains + n_grains * (n_grains - 1) / 2},
           {"M", FieldM, 1},
           {"dM", FieldDM, 2 + n_grains},
           {"kappa", FieldKappa, 2},
           {"L", FieldL, 1},
           {"energy", FieldF, 2},
           {"flux", FieldFlux, 4 * dim}}};

      // A better design is possible, but at the moment this is sufficient
      std::array<bool, n_data_variants> entries_mask;
      entries_mask.fill(false);

      unsigned int n_entries = 0;

      for (unsigned int i = 0; i < possible_entries.size(); i++)
        {
          const auto &entry = possible_entries[i];
          if (fields_list.count(std::get<0>(entry)))
            {
              entries_mask[std::get<1>(entry)] = true;
              n_entries += std::get<2>(entry);
            }
        }

      if (n_entries == 0)
        return;

      std::vector<VectorType> data_vectors(n_entries);

      for (auto &data_vector : data_vectors)
        this->matrix_free.initialize_dof_vector(data_vector, this->dof_index);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> fe_eval_all(
        this->matrix_free, this->dof_index);
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> fe_eval(
        this->matrix_free, this->dof_index);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, 1, Number, VectorizedArrayType>
          inverse_mass_matrix(fe_eval);

      AlignedVector<VectorizedArrayType> buffer(fe_eval.n_q_points * n_entries);

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto &L           = mobility.Lgb();

      vec.update_ghost_values();

      std::vector<VectorizedArrayType> temp(n_entries);

      for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
           ++cell)
        {
          fe_eval_all.reinit(cell);
          fe_eval.reinit(cell);

          fe_eval_all.reinit(cell);
          fe_eval_all.read_dof_values_plain(vec);
          fe_eval_all.evaluate(EvaluationFlags::values |
                               EvaluationFlags::gradients);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              const auto val  = fe_eval_all.get_value(q);
              const auto grad = fe_eval_all.get_gradient(q);

              const auto &c       = val[0];
              const auto &c_grad  = grad[0];
              const auto &mu_grad = grad[1];

              std::array<VectorizedArrayType, n_grains> etas;
              std::array<Tensor<1, dim, VectorizedArrayType>, n_grains>
                etas_grad;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]      = val[2 + ig];
                  etas_grad[ig] = grad[2 + ig];
                }

              unsigned int counter = 0;

              if (entries_mask[FieldBnds])
                {
                  temp[counter++] = PowerHelper<n_grains, 2>::power_sum(etas);
                }

              if (entries_mask[FieldDt])
                {
                  temp[counter++] =
                    VectorizedArrayType(data.time_data.get_current_dt());
                }

              if (entries_mask[FieldD2f])
                {
                  temp[counter++] = free_energy.d2f_dc2(c, etas);

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy.d2f_dcdetai(c, etas, ig);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy.d2f_detai2(c, etas, ig);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                        {
                          temp[counter++] =
                            free_energy.d2f_detaidetaj(c, etas, ig, jg);
                        }
                    }
                }

              if constexpr (SinteringOperatorData<dim, VectorizedArrayType>::
                              use_tensorial_mobility == false)
                {
                  if (entries_mask[FieldM])
                    {
                      temp[counter++] =
                        mobility.M(c, etas, n_grains, c_grad, etas_grad);
                    }

                  if (entries_mask[FieldDM])
                    {
                      temp[counter++] =
                        (mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad)
                          .norm();
                      temp[counter++] =
                        (mobility.dM_dgrad_c(c, c_grad, mu_grad)).norm();

                      for (unsigned int ig = 0; ig < n_grains; ++ig)
                        {
                          temp[counter++] =
                            (mobility.dM_detai(
                               c, etas, n_grains, c_grad, etas_grad, ig) *
                             mu_grad)
                              .norm();
                        }
                    }
                }
              else
                {
                  AssertThrow(entries_mask[FieldM] == false,
                              ExcNotImplemented());
                  AssertThrow(entries_mask[FieldDM] == false,
                              ExcNotImplemented());
                }

              if (entries_mask[FieldKappa])
                {
                  temp[counter++] = VectorizedArrayType(kappa_c);
                  temp[counter++] = VectorizedArrayType(kappa_p);
                }

              if (entries_mask[FieldL])
                {
                  temp[counter++] = VectorizedArrayType(L);
                }

              if (entries_mask[FieldF])
                {
                  temp[counter++] = free_energy.f(c, etas);
                  temp[counter++] = free_energy.df_dc(c, etas);
                }

              if (entries_mask[FieldFlux])
                {
                  auto j_vol  = -1. * mobility.M_vol(c) * mu_grad;
                  auto j_vap  = -1. * mobility.M_vap(c) * mu_grad;
                  auto j_surf = -1. * mobility.M_surf(c, c_grad) * mu_grad;
                  auto j_gb =
                    -1. * mobility.M_gb(etas, n_grains, etas_grad) * mu_grad;

                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      temp[counter + 0 * dim + i] = j_vol[i];
                      temp[counter + 1 * dim + i] = j_vap[i];
                      temp[counter + 2 * dim + i] = j_surf[i];
                      temp[counter + 3 * dim + i] = j_gb[i];
                    }

                  counter += 4 * dim;
                }

              for (unsigned int c = 0; c < n_entries; ++c)
                buffer[c * fe_eval.n_q_points + q] = temp[c];
            }

          for (unsigned int c = 0; c < n_entries; ++c)
            {
              inverse_mass_matrix.transform_from_q_points_to_basis(
                1,
                buffer.data() + c * fe_eval.n_q_points,
                fe_eval.begin_dof_values());

              fe_eval.set_dof_values_plain(data_vectors[c]);
            }
        }

      // TODO: remove once FEEvaluation::set_dof_values_plain()
      // sets the values of constrainging DoFs in the case of PBC
      for (unsigned int c = 0; c < n_entries; ++c)
        this->constraints.distribute(data_vectors[c]);

      vec.zero_out_ghost_values();

      // Write names of fields
      std::vector<std::string> names;
      if (entries_mask[FieldBnds])
        {
          names.push_back("bnds");
        }

      if (entries_mask[FieldDt])
        {
          names.push_back("dt");
        }

      if (entries_mask[FieldD2f])
        {
          names.push_back("d2f_dc2");

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("d2f_dcdeta" + std::to_string(ig));
            }

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("d2f_deta" + std::to_string(ig) + "2");
            }

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                {
                  names.push_back("d2f_deta" + std::to_string(ig) + "deta" +
                                  std::to_string(jg));
                }
            }
        }

      if (entries_mask[FieldM])
        {
          names.push_back("M");
        }

      if (entries_mask[FieldDM])
        {
          names.push_back("nrm_dM_dc_x_mu_grad");
          names.push_back("nrm_dM_dgrad_c");

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("nrm_dM_deta" + std::to_string(ig) +
                              "_x_mu_grad");
            }
        }

      if (entries_mask[FieldKappa])
        {
          names.push_back("kappa_c");
          names.push_back("kappa_p");
        }

      if (entries_mask[FieldL])
        {
          names.push_back("L");
        }

      if (entries_mask[FieldF])
        {
          names.push_back("f");
          names.push_back("df_dc");
        }

      if (entries_mask[FieldFlux])
        {
          std::vector fluxes{"flux_vol", "flux_vap", "flux_surf", "flux_gb"};

          for (const auto &flux_name : fluxes)
            for (unsigned int i = 0; i < dim; ++i)
              names.push_back(flux_name);
        }

      // Add data to output
      for (unsigned int c = 0; c < n_entries; ++c)
        {
          data_out.add_data_vector(this->matrix_free.get_dof_handler(
                                     this->dof_index),
                                   data_vectors[c],
                                   names[c]);
        }
    }

    void
    sanity_check(BlockVectorType &solution) const
    {
      for (unsigned int b = 0; b < solution.n_blocks(); ++b)
        if (b != 1) // If not chemical potential
          for (auto &val : solution.block(b))
            {
              if (val < 0.)
                val = 0.;
              else if (val > 1.)
                val = 1.;
            }
    }

    unsigned int
    n_components() const override
    {
      return data.n_components();
    }

    unsigned int
    n_grains() const
    {
      return this->n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      AssertDimension(n_comp - 2, n_grains);

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
      if (advection.enabled())
        advection.reinit(cell,
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
              advection.enabled())
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

              if (advection.enabled() && advection.has_velocity(ig))
                {
                  const auto &velocity =
                    advection.get_velocity(ig, phi.quadrature_point(q));
                  const auto &velocity_derivative =
                    advection.get_velocity_derivative(ig,
                                                      phi.quadrature_point(q));

                  value_result[0] +=
                    velocity * gradient[0] + velocity_derivative * c_grad;

                  value_result[ig + 2] += velocity * gradient[ig + 2] +
                                          velocity_derivative * etas_grad[ig];
                }
            }

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
      AssertDimension(n_comp - 2, n_grains);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      auto time_phi = time_integrator.create_cell_intergator(phi);

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto &order       = this->data.time_data.get_order();
      const auto &L           = mobility.Lgb();

      const auto old_solutions = history.get_old_solutions();

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
          if (advection.enabled())
            advection.reinit(cell,
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
                time_integrator.compute_time_derivative(
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
                    time_integrator.compute_time_derivative(
                      value_result[2 + ig], val, time_phi, 2 + ig, q);

                  gradient_result[2 + ig] = L * kappa_p * grad[2 + ig];

                  if (advection.enabled() && advection.has_velocity(ig))
                    {
                      const auto &velocity =
                        advection.get_velocity(ig, phi.quadrature_point(q));

                      value_result[0] += velocity * c_grad;
                      value_result[2 + ig] += velocity * grad[2 + ig];
                    }
                }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &  data;
    const TimeIntegration::SolutionHistory<BlockVectorType> &history;
    const TimeIntegration::BDFIntegrator<dim, Number, VectorizedArrayType>
                                                                time_integrator;
    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;
  };
} // namespace Sintering