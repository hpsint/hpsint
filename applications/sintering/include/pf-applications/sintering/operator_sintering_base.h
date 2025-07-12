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

#include <deal.II/base/table_handler.h>

#include <pf-applications/sintering/free_energy.h>
#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/time_integration/solution_history.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class SinteringOperatorBase
    : public OperatorBase<dim, Number, VectorizedArrayType, T>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType>      &matrix_free,
      const AffineConstraints<Number>                         &constraints,
      const FreeEnergy<VectorizedArrayType>                   &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>   &data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &history,
      const bool                                               matrix_based)
      : OperatorBase<dim, Number, VectorizedArrayType, T>(matrix_free,
                                                          constraints,
                                                          0,
                                                          "sintering_op",
                                                          matrix_based)
      , free_energy(free_energy)
      , data(data)
      , history(history)
    {}

    ~SinteringOperatorBase()
    {}

    const SinteringOperatorData<dim, VectorizedArrayType> &
    get_data() const
    {
      return data;
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim>                &data_out,
                               const BlockVectorType       &vec,
                               const std::set<std::string> &fields_list) const
    {
      // Possible output options
      enum OutputFields
      {
        FieldBnds,
        FieldGb,
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
           {"gb", FieldGb, 1},
           {"d2f", FieldD2f, 1 + 2 * n_grains + n_grains * (n_grains - 1) / 2},
           {"M", FieldM, 1},
           {"dM", FieldDM, 2 + n_grains},
           {"kappa", FieldKappa, 2},
           {"L", FieldL, 1},
           {"energy", FieldF, 2},
           {"flux", FieldFlux, 4 * dim}}};

      // Get active entries to output
      const auto [entries_mask, n_entries] =
        this->get_vector_output_entries_mask(possible_entries, fields_list);

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

      const auto &free_energy = this->free_energy;
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

              if (entries_mask[FieldGb])
                {
                  VectorizedArrayType etaijSum = 0.0;
                  for (unsigned int i = 0; i < n_grains; ++i)
                    for (unsigned int j = 0; j < i; ++j)
                      etaijSum += etas[i] * etas[j];

                  temp[counter++] = etaijSum;
                }

              const auto free_energy_eval =
                free_energy.template eval<EnergyAll, n_grains>(val);

              if (entries_mask[FieldD2f])
                {
                  temp[counter++] = free_energy_eval.d2f_dc2();

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy_eval.d2f_dcdetai(etas[ig]);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy_eval.d2f_detai2(etas[ig]);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                        {
                          temp[counter++] =
                            free_energy_eval.d2f_detaidetaj(etas[ig], etas[jg]);
                        }
                    }
                }

              if constexpr (SinteringOperatorData<dim, VectorizedArrayType>::
                              use_tensorial_mobility == false)
                {
                  if (entries_mask[FieldM])
                    {
                      Tensor<1, dim, VectorizedArrayType> dummy;
                      dummy[0] = 1.0;

                      temp[counter++] = mobility.apply_M(
                        c, etas, n_grains, c_grad, etas_grad, dummy)[0];
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
                  temp[counter++] = free_energy_eval.f();
                  temp[counter++] = free_energy_eval.df_dc();
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

      if (entries_mask[FieldGb])
        {
          names.push_back("gb");
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
          data_out.add_data_vector(data_vectors[c], names[c]);
        }
    }

    /* Build scalar quantities to compute */
    auto
    build_domain_quantities_evaluators(
      const std::vector<std::string> &labels) const
    {
      using QuantityCallback = std::function<
        VectorizedArrayType(const VectorizedArrayType *,
                            const Tensor<1, dim, VectorizedArrayType> *)>;

      std::vector<std::string>      q_labels;
      std::vector<QuantityCallback> q_evaluators;

      for (const auto &qty : labels)
        {
          if (qty == "solid_vol")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType                 *value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                return value[0];
              });
          else if (qty == "surf_area")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType                 *value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                return value[0] * (1.0 - value[0]);
              });
          else if (qty == "gb_area")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                VectorizedArrayType eta_ij_sum = 0.0;
                for (unsigned int i = 0; i < data.n_grains(); ++i)
                  for (unsigned int j = i + 1; j < data.n_grains(); ++j)
                    eta_ij_sum += value[2 + i] * value[2 + j];

                return eta_ij_sum;
              });
          else if (qty == "avg_grain_size")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                VectorizedArrayType eta_i2_sum = 0.0;
                for (unsigned int i = 0; i < data.n_grains(); ++i)
                  eta_i2_sum += value[2 + i] * value[2 + i];

                return eta_i2_sum;
              });
          else if (qty == "surf_area_nrm")
            q_evaluators.emplace_back(
              [](const VectorizedArrayType                 *value,
                 const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                VectorizedArrayType c_int(1.0);
                c_int = compare_and_apply_mask<SIMDComparison::less_than>(
                  value[0],
                  VectorizedArrayType(0.45),
                  VectorizedArrayType(0.0),
                  c_int);
                c_int = compare_and_apply_mask<SIMDComparison::greater_than>(
                  value[0],
                  VectorizedArrayType(0.55),
                  VectorizedArrayType(0.0),
                  c_int);

                return c_int;
              });
          else if (qty == "free_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                VectorizedArrayType energy(0.0);

                for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
                  energy += gradient[2 + ig].norm_square();

                energy *= 0.5 * data.kappa_p;

                const auto &c_grad = gradient[0];
                energy += 0.5 * data.kappa_c * c_grad.norm_square();

                const auto free_energy_eval =
                  free_energy.template eval<EnergyZero>(value, data.n_grains());

                energy += free_energy_eval.f();

                return energy;
              });
          else if (qty == "bulk_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                const auto free_energy_eval =
                  free_energy.template eval<EnergyZero>(value, data.n_grains());

                return free_energy_eval.f();
              });
          else if (qty == "interface_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)value;

                VectorizedArrayType energy(0.0);

                for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
                  energy += gradient[2 + ig].norm_square();
                energy *= 0.5 * data.kappa_p;

                const auto &c_grad = gradient[0];
                energy += 0.5 * data.kappa_c * c_grad.norm_square();

                return energy;
              });
          else if (qty == "order_params")
            for (unsigned int i = 0; i < MAX_SINTERING_GRAINS; ++i)
              {
                // The number of order parameters can vary so we will output the
                // maximum number of them. The unused order parameters will be
                // simply filled with zeros.
                q_labels.push_back("op_" + std::to_string(i));

                q_evaluators.emplace_back(
                  [this,
                   i](const VectorizedArrayType                 *value,
                      const Tensor<1, dim, VectorizedArrayType> *gradient) {
                    (void)gradient;

                    return i < data.n_grains() ? value[2 + i] : 0.;
                  });
              }
          else if (qty == "control_vol")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType                 *value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)value;
                (void)gradient;

                return VectorizedArrayType(1.);
              });
          else
            AssertThrow(false,
                        ExcMessage("Invalid domain integral provided: " + qty));

          if (qty != "order_params")
            q_labels.push_back(qty);
        }

      AssertDimension(q_labels.size(), q_evaluators.size());

      return std::make_tuple(q_labels, q_evaluators);
    }

    const FreeEnergy<VectorizedArrayType> &
    get_free_energy() const
    {
      return free_energy;
    }

  protected:
    const FreeEnergy<VectorizedArrayType>                    free_energy;
    const SinteringOperatorData<dim, VectorizedArrayType>   &data;
    const TimeIntegration::SolutionHistory<BlockVectorType> &history;
  };
} // namespace Sintering
