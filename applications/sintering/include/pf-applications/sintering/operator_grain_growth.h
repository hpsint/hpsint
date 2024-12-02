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

#include <deal.II/base/table_handler.h>

#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <pf-applications/base/mask.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/numerics/power_helper.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/time_integration/solution_history.h>

namespace Sintering
{
  using namespace dealii;

  template <typename VectorizedArrayType>
  class GrainGrowthFreeEnergy
  {
  private:
    double A;
    double B;

    // This class knows about the structure of the state vector
    template <bool with_power_4>
    class Evaluation
    {
    public:
      template <int n_grains>
      Evaluation(const double               A,
                 const double               B,
                 const VectorizedArrayType *state,
                 std::integral_constant<int, n_grains>)
        : A(A)
        , B(B)
      {
        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(state);

        if constexpr (with_power_4)
          {
            etaPower4Sum = PowerHelper<n_grains, 4>::power_sum(state);
            mixed_sum    = 0;
            for (unsigned int i = 0; i < n_grains; ++i)
              for (unsigned int j = i + 1; j < n_grains; ++j)
                mixed_sum += state[i] * state[j];
          }
      }

      template <typename VectorType, int n_grains>
      Evaluation(const double      A,
                 const double      B,
                 const VectorType &state,
                 std::integral_constant<int, n_grains>)
        : A(A)
        , B(B)
      {
        std::array<VectorizedArrayType, n_grains> etas;
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          etas[ig] = state[ig];

        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(etas);

        if constexpr (with_power_4)
          {
            etaPower4Sum = PowerHelper<n_grains, 4>::power_sum(etas);
            mixed_sum    = 0;
            for (unsigned int i = 0; i < n_grains; ++i)
              for (unsigned int j = i + 1; j < n_grains; ++j)
                mixed_sum += etas[i] * etas[j];
          }
      }

      template <typename VectorType>
      Evaluation(const double       A,
                 const double       B,
                 const VectorType & state,
                 const unsigned int n_grains)
        : A(A)
        , B(B)
      {
        (void)n_grains;

        std::vector<VectorizedArrayType> etas(n_grains);
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          etas[ig] = state[ig];

        etaPower2Sum = PowerHelper<0, 2>::power_sum(etas);

        if constexpr (with_power_4)
          {
            etaPower4Sum = PowerHelper<0, 4>::power_sum(etas);
            mixed_sum    = 0;
            for (unsigned int i = 0; i < n_grains; ++i)
              for (unsigned int j = i + 1; j < n_grains; ++j)
                mixed_sum += etas[i] * etas[j];
          }
      }

      template <typename VectorType,
                int n_grains = SizeHelper<VectorType>::size>
      Evaluation(const double A, const double B, const VectorType &state)
        : A(A)
        , B(B)
      {
        etaPower2Sum = PowerHelper<n_grains, 2>::power_sum(state);

        if constexpr (with_power_4)
          {
            etaPower4Sum = PowerHelper<n_grains, 4>::power_sum(state);
            mixed_sum    = 0;
            for (unsigned int i = 0; i < n_grains; ++i)
              for (unsigned int j = i + 1; j < n_grains; ++j)
                mixed_sum += state[i] * state[j];
          }
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      f() const
      {
        Assert(with_power_4,
               ExcMessage("The evaluator was initialized without "
                          " parameter with_power_4 enabled"));

        return -A / 2. * etaPower2Sum + B / 4. * etaPower4Sum + mixed_sum;
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      df_detai(const VectorizedArrayType &etai) const
      {
        const auto etai2 = etai * etai;
        const auto etai3 = etai * etai * etai;

        return -A * etai + B * etai3 + 2. * etai * (etaPower2Sum - etai2);
      }

      // TODO: remove this, has been kept to instantiate block preconditioners
      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_dc2() const
      {
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detai2(const VectorizedArrayType &etai) const
      {
        const auto etai2 = etai * etai;

        return -A + 3. * B * etai2 + 2. * (etaPower2Sum - etai2);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detaidetaj(const VectorizedArrayType &etai,
                     const VectorizedArrayType &etaj) const
      {
        return 4. * etai * etaj;
      }

    private:
      double              A;
      double              B;
      VectorizedArrayType mixed_sum;
      VectorizedArrayType etaPower2Sum;
      VectorizedArrayType etaPower4Sum;
    };

    template <typename Mask>
    using EvaluationConcrete =
      Evaluation<any_energy_eval_of_v<Mask, EnergyEvaluation::zero>>;

  public:
    static const unsigned int op_components_offset = 0;

    GrainGrowthFreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <typename Mask, int n_grains>
    EvaluationConcrete<Mask>
    eval(const VectorizedArrayType *state) const
    {
      return EvaluationConcrete<Mask>(A,
                                      B,
                                      state,
                                      std::integral_constant<int, n_grains>());
    }

    template <typename Mask>
    EvaluationConcrete<Mask>
    eval(const VectorizedArrayType *state, const unsigned int n_grains) const
    {
      return EvaluationConcrete<Mask>(A, B, state, n_grains);
    }

    template <typename Mask, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state, const unsigned int n_grains) const
    {
      return EvaluationConcrete<Mask>(A, B, state, n_grains);
    }

    template <typename Mask, int n_grains, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state) const
    {
      return EvaluationConcrete<Mask>(A,
                                      B,
                                      state,
                                      std::integral_constant<int, n_grains>());
    }

    template <typename Mask, typename VectorType>
    EvaluationConcrete<Mask>
    eval(const VectorType &state) const
    {
      return EvaluationConcrete<Mask>(A, B, state);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE void
    apply_d2f_detaidetaj(const VectorType &         L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType *      value_result) const
    {
      if (n_grains <= 1)
        return; // nothing to do

      VectorizedArrayType temp;

      for (unsigned int ig = 1; ig < n_grains; ++ig)
        temp += etas[ig] * value[ig];

      for (unsigned int ig = 0; ig < n_grains; ++ig)
        value_result[ig] +=
          (L * 4.0) * etas[ig] * (temp - etas[ig] * value[ig]);
    }
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class GrainGrowthOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          GrainGrowthOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = GrainGrowthOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    GrainGrowthOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const GrainGrowthFreeEnergy<VectorizedArrayType> &       free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType> &  data,
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
      , time_integrator(data.time_data, history)
    {}

    ~GrainGrowthOperator()
    {}

    template <typename... Arg>
    static T
    create(
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const GrainGrowthFreeEnergy<VectorizedArrayType> &       free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType> &  sintering_data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &solution_history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
        &        advection_mechanism,
      const bool matrix_based,
      Arg &&...)
    {
      (void)advection_mechanism;

      return T(matrix_free,
               constraints,
               free_energy,
               sintering_data,
               solution_history,
               matrix_based);
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
      // Possible output options
      enum OutputFields
      {
        FieldBnds,
        FieldGb,
        FieldF,
        FieldDf,
        FieldD2f
      };

      constexpr unsigned int n_data_variants = 5;

      const std::array<std::tuple<std::string, OutputFields, unsigned int>,
                       n_data_variants>
        possible_entries = {{{"bnds", FieldBnds, 1},
                             {"gb", FieldGb, 1},
                             {"energy", FieldF, 1},
                             {"df", FieldD2f, n_grains},
                             {"d2f", FieldD2f, n_grains * (n_grains - 1) / 2}}};

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
              const auto etas = fe_eval_all.get_value(q);

              unsigned int counter = 0;

              if (entries_mask[FieldBnds])
                temp[counter++] = PowerHelper<n_grains, 2>::power_sum(etas);

              if (entries_mask[FieldGb])
                {
                  VectorizedArrayType etaijSum = 0.0;
                  for (unsigned int i = 0; i < n_grains; ++i)
                    for (unsigned int j = 0; j < i; ++j)
                      etaijSum += etas[i] * etas[j];

                  temp[counter++] = etaijSum;
                }

              const auto free_energy_eval =
                free_energy.template eval<EnergyAll, n_grains>(etas);

              if (entries_mask[FieldF])
                temp[counter++] = free_energy_eval.f();

              if (entries_mask[FieldDf])
                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  temp[counter++] = free_energy_eval.df_detai(etas[ig]);

              if (entries_mask[FieldD2f])
                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                    temp[counter++] =
                      free_energy_eval.d2f_detaidetaj(etas[ig], etas[jg]);

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
        names.push_back("bnds");

      if (entries_mask[FieldGb])
        names.push_back("gb");

      if (entries_mask[FieldF])
        names.push_back("f");

      if (entries_mask[FieldD2f])
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          names.push_back("df_deta" + std::to_string(ig));

      if (entries_mask[FieldD2f])
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          {
            names.push_back("d2f_deta" + std::to_string(ig) + "2");

            for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
              names.push_back("d2f_deta" + std::to_string(ig) + "deta" +
                              std::to_string(jg));
          }

      // Add data to output
      for (unsigned int c = 0; c < n_entries; ++c)
        {
          data_out.add_data_vector(data_vectors[c], names[c]);
        }
    }

    unsigned int
    n_components() const override
    {
      return this->data.n_components();
    }

    unsigned int
    n_grains() const
    {
      return this->data.n_components();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains;
    }

    template <int with_time_derivative>
    void
    evaluate_nonlinear_residual(
      BlockVectorType &      dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

      std::function<void(const unsigned int, const unsigned int)>
        pre_operation_to_be_used = pre_operation;

      if (!pre_operation)
        // no pre-function is given -> attach function to clear dst vector
        pre_operation_to_be_used = [&](const auto start_range,
                                       const auto end_range) {
          for (unsigned int b = 0; b < internal::n_blocks(dst); ++b)

            std::memset(internal::block(dst, b).begin() + start_range,
                        0,
                        sizeof(Number) * (end_range - start_range));
        };

#define OPERATION(c, d)                                           \
  MyMatrixFreeTools::cell_loop_wrapper(                           \
    this->matrix_free,                                            \
    &GrainGrowthOperator::                                        \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    pre_operation_to_be_used,                                     \
    post_operation);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      AssertDimension(n_comp, n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values = this->data.get_nonlinear_values();

      const auto &free_energy = this->free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_p     = this->data.kappa_p;
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &L           = mobility.Lgb();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          typename FECellIntegratorType::value_type    value_result;
          typename FECellIntegratorType::gradient_type gradient_result;

          const auto  value     = phi.get_value(q);
          const auto  gradient  = phi.get_gradient(q);
          const auto &lin_value = nonlinear_values[cell][q];

          const auto free_energy_eval =
            free_energy.template eval<EnergySecond, n_grains>(lin_value);

          // process eta rows
          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[ig] +=
                value[ig] * weight +
                L * free_energy_eval.d2f_detai2(lin_value[ig]) * value[ig];

              gradient_result[ig] = L * kappa_p * gradient[ig];

              for (unsigned int jg = 0; jg < ig; ++jg)
                {
                  const auto d2f_detaidetaj =
                    free_energy_eval.d2f_detaidetaj(lin_value[ig],
                                                    lin_value[jg]);

                  value_result[ig] += L * d2f_detaidetaj * value[jg];
                  value_result[jg] += L * d2f_detaidetaj * value[ig];
                }
            }

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    template <typename FECellIntegratorType>
    void
    precondition_advection_ch(
      const unsigned int q,
      const AdvectionVelocityData<dim, Number, VectorizedArrayType>
        &                                                     advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType &                            phi,
      typename FECellIntegratorType::value_type &             value_result,
      typename FECellIntegratorType::gradient_type &gradient_result) const
    {
      (void)q;
      (void)advection_data;
      (void)nonlinear_data;
      (void)phi;
      (void)value_result;
      (void)gradient_result;
    }

    template <typename FECellIntegratorType>
    void
    precondition_advection_ac(
      const unsigned int q,
      const AdvectionVelocityData<dim, Number, VectorizedArrayType>
        &                                                     advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType &                            phi,
      typename FECellIntegratorType::value_type &             value_result,
      typename FECellIntegratorType::gradient_type &gradient_result) const
    {
      (void)q;
      (void)advection_data;
      (void)nonlinear_data;
      (void)phi;
      (void)value_result;
      (void)gradient_result;
    }

    template <typename FECellIntegratorType>
    void
    precondition_advection_ac(
      const unsigned int q,
      const unsigned int igrain,
      const AdvectionVelocityData<dim, Number, VectorizedArrayType>
        &                                                     advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType &                            phi,
      typename FECellIntegratorType::value_type &             value_result,
      typename FECellIntegratorType::gradient_type &gradient_result) const
    {
      (void)q;
      (void)igrain;
      (void)advection_data;
      (void)nonlinear_data;
      (void)phi;
      (void)value_result;
      (void)gradient_result;
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
          if (qty == "gb_area")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                VectorizedArrayType eta_ij_sum = 0.0;
                for (unsigned int i = 0; i < data.n_grains(); ++i)
                  for (unsigned int j = i + 1; j < data.n_grains(); ++j)
                    eta_ij_sum += value[i] * value[j];

                return eta_ij_sum;
              });
          else if (qty == "avg_grain_size")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                VectorizedArrayType eta_i2_sum = 0.0;
                for (unsigned int i = 0; i < data.n_grains(); ++i)
                  eta_i2_sum += value[i] * value[i];

                return eta_i2_sum;
              });
          else if (qty == "free_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                VectorizedArrayType energy(0.0);

                for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
                  energy += gradient[ig].norm_square();

                energy *= 0.5 * data.kappa_p;

                const auto free_energy_eval =
                  free_energy.template eval<EnergyZero>(value, data.n_grains());

                energy += free_energy_eval.f();

                return energy;
              });
          else if (qty == "bulk_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)gradient;

                const auto free_energy_eval =
                  free_energy.template eval<EnergyZero>(value, data.n_grains());

                return free_energy_eval.f();
              });
          else if (qty == "interface_energy")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
                     const Tensor<1, dim, VectorizedArrayType> *gradient) {
                (void)value;

                VectorizedArrayType energy(0.0);

                for (unsigned int ig = 0; ig < data.n_grains(); ++ig)
                  energy += gradient[ig].norm_square();
                energy *= 0.5 * data.kappa_p;

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
                   i](const VectorizedArrayType *                value,
                      const Tensor<1, dim, VectorizedArrayType> *gradient) {
                    (void)gradient;

                    return i < data.n_grains() ? value[i] : 0.;
                  });
              }
          else if (qty == "control_vol")
            q_evaluators.emplace_back(
              [this](const VectorizedArrayType *                value,
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

    const GrainGrowthFreeEnergy<VectorizedArrayType> &
    get_free_energy() const
    {
      return free_energy;
    }

  private:
    template <int n_comp, int n_grains, int with_time_derivative>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      AssertDimension(n_comp, n_grains);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      auto time_phi = this->time_integrator.create_cell_intergator(phi);

      const auto &mobility = this->data.get_mobility();
      const auto &kappa_p  = this->data.kappa_p;
      const auto &order    = this->data.time_data.get_order();
      const auto &L        = mobility.Lgb();

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
              const auto etas      = phi.get_value(q);
              const auto etas_grad = phi.get_gradient(q);

              const auto free_energy_eval =
                free_energy.template eval<EnergyFirst, n_grains>(etas);

              typename FECellIntegrator<dim,
                                        n_comp,
                                        Number,
                                        VectorizedArrayType>::value_type
                value_result;
              typename FECellIntegrator<dim,
                                        n_comp,
                                        Number,
                                        VectorizedArrayType>::gradient_type
                gradient_result;

              // AC equations
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[ig] = L * free_energy_eval.df_detai(etas[ig]);

                  if (with_time_derivative)
                    this->time_integrator.compute_time_derivative(
                      value_result[ig], etas, time_phi, ig, q);

                  gradient_result[ig] = L * kappa_p * etas_grad[ig];
                }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

  protected:
    const GrainGrowthFreeEnergy<VectorizedArrayType>         free_energy;
    const SinteringOperatorData<dim, VectorizedArrayType> &  data;
    const TimeIntegration::SolutionHistory<BlockVectorType> &history;
    const TimeIntegration::BDFIntegrator<dim, Number, VectorizedArrayType>
      time_integrator;
  };
} // namespace Sintering
