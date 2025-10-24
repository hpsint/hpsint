// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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
  class GreenquistFreeEnergy
  {
  private:
    using Number = double;

    // This class knows about the structure of the state vector
    class Evaluation
    {
    public:
      template <int n_grains>
      Evaluation(const VectorizedArrayType *state,
                 std::integral_constant<int, n_grains>)
      {
        std::array<VectorizedArrayType, n_grains + 1> phases;
        phases[0] = state[0];
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          phases[ig + 1] = state[ig + 2];

        phasesPower2Sum = PowerHelper<n_grains + 1, 2>::power_sum(phases);
        phasesPower4Sum = PowerHelper<n_grains + 1, 4>::power_sum(phases);

        mixed_sum2 = 0;
        for (unsigned int i = 0; i < phases.size(); ++i)
          for (unsigned int j = i + 1; j < phases.size(); ++j)
            mixed_sum2 += phases[i] * phases[i] * phases[j] * phases[j];
        mixed_sum2 *= 2.;

        phi = state[0];
        mu  = state[1];
      }

      template <typename VectorType, int n_grains>
      Evaluation(const VectorType &state, std::integral_constant<int, n_grains>)
      {
        std::array<VectorizedArrayType, n_grains + 1> phases;
        phases[0] = state[0];
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          phases[ig + 1] = state[ig + 2];

        phasesPower2Sum = PowerHelper<n_grains + 1, 2>::power_sum(phases);
        phasesPower4Sum = PowerHelper<n_grains + 1, 4>::power_sum(phases);

        mixed_sum2 = 0;
        for (unsigned int i = 0; i < phases.size(); ++i)
          for (unsigned int j = i + 1; j < phases.size(); ++j)
            mixed_sum2 += phases[i] * phases[i] * phases[j] * phases[j];
        mixed_sum2 *= 2.;

        phi = state[0];
        mu  = state[1];
      }

      template <typename VectorType>
      Evaluation(const VectorType &state, const unsigned int n_grains)
      {
        std::vector<VectorizedArrayType> phases(n_grains + 1);
        phases[0] = state[0];
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          phases[ig + 1] = state[ig + 2];

        phasesPower2Sum = PowerHelper<0, 2>::power_sum(phases);
        phasesPower4Sum = PowerHelper<0, 4>::power_sum(phases);

        mixed_sum2 = 0;
        for (unsigned int i = 0; i < phases.size(); ++i)
          for (unsigned int j = i + 1; j < phases.size(); ++j)
            mixed_sum2 += phases[i] * phases[i] * phases[j] * phases[j];
        mixed_sum2 *= 2.;

        phi = state[0];
        mu  = state[1];
      }

      template <typename VectorType, int n_comp = SizeHelper<VectorType>::size>
      Evaluation(const VectorType &state)
      {
        std::array<VectorizedArrayType, n_comp - 1> phases;
        phases[0] = state[0];
        for (unsigned int ig = 0; ig < n_comp - 2; ++ig)
          phases[ig + 1] = state[ig + 2];

        phasesPower2Sum = PowerHelper<n_comp - 1, 2>::power_sum(phases);
        phasesPower4Sum = PowerHelper<n_comp - 1, 4>::power_sum(phases);

        mixed_sum2 = 0;
        for (unsigned int i = 0; i < phases.size(); ++i)
          for (unsigned int j = i + 1; j < phases.size(); ++j)
            mixed_sum2 += phases[i] * phases[i] * phases[j] * phases[j];
        mixed_sum2 *= 2.;

        phi = state[0];
        mu  = state[1];
      }

      VectorizedArrayType
      H(const VectorizedArrayType &phi0) const
      {
        VectorizedArrayType h = 6. * Utilities::fixed_power<5>(phi) /
                                  Utilities::fixed_power<5>(phi0) -
                                15. * Utilities::fixed_power<4>(phi) /
                                  Utilities::fixed_power<4>(phi0) +
                                10. * Utilities::fixed_power<3>(phi) /
                                  Utilities::fixed_power<3>(phi0);

        h = compare_and_apply_mask<SIMDComparison::less_than>(
          phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), h);
        h = compare_and_apply_mask<SIMDComparison::greater_than>(
          phi, phi0, VectorizedArrayType(1.0), h);

        return h;
      }

      VectorizedArrayType
      dH(const VectorizedArrayType &phi0) const
      {
        VectorizedArrayType h = 30. * Utilities::fixed_power<4>(phi) /
                                  Utilities::fixed_power<5>(phi0) -
                                60. * Utilities::fixed_power<3>(phi) /
                                  Utilities::fixed_power<4>(phi0) +
                                30. * Utilities::fixed_power<2>(phi) /
                                  Utilities::fixed_power<3>(phi0);

        h = compare_and_apply_mask<SIMDComparison::less_than>(
          phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), h);
        h = compare_and_apply_mask<SIMDComparison::greater_than>(
          phi, phi0, VectorizedArrayType(0.0), h);

        return h;
      }

      VectorizedArrayType
      d2H(const VectorizedArrayType &phi0) const
      {
        VectorizedArrayType h = 120. * Utilities::fixed_power<3>(phi) /
                                  Utilities::fixed_power<5>(phi0) -
                                180. * Utilities::fixed_power<2>(phi) /
                                  Utilities::fixed_power<4>(phi0) +
                                60. * phi / Utilities::fixed_power<3>(phi0);

        h = compare_and_apply_mask<SIMDComparison::less_than>(
          phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), h);
        h = compare_and_apply_mask<SIMDComparison::greater_than>(
          phi, phi0, VectorizedArrayType(0.0), h);

        return h;
      }

      VectorizedArrayType
      hb() const
      {
        return H(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      dhb() const
      {
        return dH(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      d2hb() const
      {
        return d2H(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      hm() const
      {
        return VectorizedArrayType(1.0) - H(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      dhm() const
      {
        return -dH(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      d2hm() const
      {
        return -d2H(VectorizedArrayType(1.0));
      }

      VectorizedArrayType
      chi() const
      {
        return VectorizedArrayType(1.) / (Va * Va) * (hb() / kb + hm() / km);
      }

      VectorizedArrayType
      dchi_dphi() const
      {
        return VectorizedArrayType(1.) / (Va * Va) * (dhb() / kb + dhm() / km);
      }

      VectorizedArrayType
      switching(const Number fac_s, const Number fac_gb) const
      {
        VectorizedArrayType phi0(0.3);
        return fac_s * H(phi0) + fac_gb * (VectorizedArrayType(1.) - H(phi0));
      }

      VectorizedArrayType
      dswitching_dphi(const Number fac_s, const Number fac_gb) const
      {
        VectorizedArrayType phi0(0.3);
        return fac_s * dH(phi0) - fac_gb * dH(phi0);
      }

      VectorizedArrayType
      d2switching_dphi2(const Number fac_s, const Number fac_gb) const
      {
        VectorizedArrayType phi0(0.3);
        return fac_s * d2H(phi0) - fac_gb * d2H(phi0);
      }

      VectorizedArrayType
      epsilon() const
      {
        return switching(eps_s, eps_gb);
      }

      VectorizedArrayType
      depsilon_dphi() const
      {
        return dswitching_dphi(eps_s, eps_gb);
      }

      VectorizedArrayType
      d2epsilon_dphi2() const
      {
        return d2switching_dphi2(eps_s, eps_gb);
      }

      VectorizedArrayType
      kappa() const
      {
        return switching(kappa_s, kappa_gb);
      }

      VectorizedArrayType
      dkappa_dphi() const
      {
        return dswitching_dphi(kappa_s, kappa_gb);
      }

      VectorizedArrayType
      d2kappa_dphi2() const
      {
        return d2switching_dphi2(kappa_s, kappa_gb);
      }

      // Omega_b
      VectorizedArrayType
      omega_b() const
      {
        return -1. * mu * mu / (2 * Va * Va * kb) - cb_eq * mu / Va;
      }

      VectorizedArrayType
      domega_b_dmu() const
      {
        return -1. * mu / (Va * Va * kb) - VectorizedArrayType(cb_eq / Va);
      }

      VectorizedArrayType
      d2omega_b_dmu2() const
      {
        return VectorizedArrayType(-1. / (Va * Va * kb));
      }

      // Omega_m
      VectorizedArrayType
      omega_m() const
      {
        return -1. * mu * mu / (2 * Va * Va * km) - cm_eq() * mu / Va;
      }

      VectorizedArrayType
      domega_m_dmu() const
      {
        return -1. * mu / (Va * Va * km) - cm_eq() / Va;
      }

      VectorizedArrayType
      domega_m_dphasei(const VectorizedArrayType &phase_i) const
      {
        return -1. * dcm_eq_dphasei(phase_i) * mu / Va;
      }

      VectorizedArrayType
      d2omega_m_dmu2(const VectorizedArrayType &mu) const
      {
        return VectorizedArrayType(-1. / (Va * Va * km));
      }

      VectorizedArrayType
      d2omega_m_dmudphasei(const VectorizedArrayType &phase_i) const
      {
        return -1. * dcm_eq_dphasei(phase_i) / Va;
      }

      VectorizedArrayType
      d3omega_m_dmudphasei2(const VectorizedArrayType &phase_i) const
      {
        return -1. * d2cm_eq_dphasei2(phase_i) / Va;
      }

      VectorizedArrayType
      d3omega_m_dmudphaseidphasej(const VectorizedArrayType &phase_i,
                                  const VectorizedArrayType &phase_j) const
      {
        return -1. * d2cm_eq_dphaseiphasej(phase_i, phase_j) / Va;
      }

      VectorizedArrayType
      d2omega_m_dphasei2(const VectorizedArrayType &phase_i) const
      {
        return -1. * d2cm_eq_dphasei2(phase_i) * mu / Va;
      }

      VectorizedArrayType
      d2omega_m_dphaseidphasej(const VectorizedArrayType &phase_i,
                               const VectorizedArrayType &phase_j) const
      {
        return -1. * d2cm_eq_dphaseiphasej(phase_i, phase_j) * mu / Va;
      }

      // Concentrations
      VectorizedArrayType
      cm_eq() const
      {
        return ceq_B + 4. * (ceq_GB - ceq_B) *
                         Utilities::fixed_power<2>(VectorizedArrayType(1.) -
                                                   phasesPower2Sum);
      }

      VectorizedArrayType
      dcm_eq_dphasei(const VectorizedArrayType &phase_i) const
      {
        return -16. * (ceq_GB - ceq_B) *
               (VectorizedArrayType(1.) - phasesPower2Sum) * phase_i;
      }

      VectorizedArrayType
      d2cm_eq_dphasei2(const VectorizedArrayType &phase_i) const
      {
        return -16. * (ceq_GB - ceq_B) *
               (VectorizedArrayType(1.) - phasesPower2Sum -
                2. * phase_i * phase_i);
      }

      VectorizedArrayType
      d2cm_eq_dphaseiphasej(const VectorizedArrayType &phase_i,
                            const VectorizedArrayType &phase_j) const
      {
        return 32. * (ceq_GB - ceq_B) * phase_i * phase_j;
      }

      VectorizedArrayType
      zeta() const
      {
        return phasesPower4Sum + gamma * mixed_sum2 +
               VectorizedArrayType(1. / 4.);
      }

      VectorizedArrayType
      dzeta_dphasei(const VectorizedArrayType &phase_i) const
      {
        return Utilities::fixed_power<3>(phase_i) - phase_i +
               2. * gamma * phase_i *
                 (phasesPower2Sum - Utilities::fixed_power<2>(phase_i));
      }

      VectorizedArrayType
      d2zeta_dphasei2(const VectorizedArrayType &phase_i) const
      {
        return 3. * Utilities::fixed_power<2>(phase_i) -
               VectorizedArrayType(1.0) +
               2. * gamma *
                 (phasesPower2Sum - Utilities::fixed_power<2>(phase_i));
      }

      VectorizedArrayType
      d2zeta_dphaseidphasej(const VectorizedArrayType &phase_i,
                            const VectorizedArrayType &phase_j) const
      {
        return 4. * gamma * phase_i * phase_j;
      }

      Number
      get_L() const
      {
        return L;
      }

      Number
      get_Lphi() const
      {
        return L_phi;
      }

      // TODO: compute energy properly
      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      f() const
      {
        return VectorizedArrayType(0.);
      }

      // Dummy functions for preconditioners
      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      df_dc() const
      {
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      df_detai(const VectorizedArrayType &etai) const
      {
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_dc2() const
      {
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_dcdetai(const VectorizedArrayType &etai) const
      {
        (void)etai;
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detai2(const VectorizedArrayType &etai) const
      {
        (void)etai;
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detaidetaj(const VectorizedArrayType &etai,
                     const VectorizedArrayType &etaj) const
      {
        (void)etai;
        (void)etaj;
        return VectorizedArrayType(0.);
      }

    private:
      Number kB{8.617333262145e-5}; // Ev/K - Boltzman

      Number interface_width{60};

      Number T{1816};
      Number km{157.16 - 0.0025 * T};
      Number kb{10 * km};
      Number Va{0.04092};

      Number M0{1.476e9};
      Number Q{2.77};
      Number T0{2294};

      Number Mv{M0 * std::exp(-Q / (kB * T0))};
      Number Ms{M0 * std::exp(-Q / (kB * T))};

      Number L_phi{4. / 3. * Mv / interface_width};
      Number L{4. / 3. * Ms / interface_width};

      Number sigma_s{19.72};
      Number sigma_gb{9.86};
      Number eps_s{6 * sigma_s / interface_width};
      Number eps_gb{6 * sigma_gb / interface_width};
      Number kappa_s{3. / 4. * sigma_s * interface_width};
      Number kappa_gb{3. / 4. * sigma_gb * interface_width};

      Number Ef{2.69};
      Number ceq_B{std::exp(-Ef / (kB * T))};
      Number ceq_GB{0.189};
      Number cb_eq{1.0};

      Number gamma{1.5};

      VectorizedArrayType phi;
      VectorizedArrayType mu;
      VectorizedArrayType phasesPower2Sum;
      VectorizedArrayType phasesPower4Sum;
      VectorizedArrayType mixed_sum2;
    };

  public:
    static const unsigned int op_components_offset  = 2;
    static const bool         concentration_as_void = true;
    static const bool         save_gradients        = false;

    GreenquistFreeEnergy(double A, double B)
    {
      (void)A;
      (void)B;
    }

    // We still need Mask param since it is used in the preconditioners,
    // however, for this energy functional it is not used.
    template <typename Mask, int n_grains>
    Evaluation
    eval(const VectorizedArrayType *state) const
    {
      return Evaluation(state, std::integral_constant<int, n_grains>());
    }

    template <typename Mask>
    Evaluation
    eval(const VectorizedArrayType *state, const unsigned int n_grains) const
    {
      return Evaluation(state, n_grains);
    }

    template <typename Mask, typename VectorType>
    Evaluation
    eval(const VectorType &state, const unsigned int n_grains) const
    {
      return Evaluation(state, n_grains);
    }

    template <typename Mask, int n_grains, typename VectorType>
    Evaluation
    eval(const VectorType &state) const
    {
      return Evaluation(state, std::integral_constant<int, n_grains>());
    }

    template <typename Mask, typename VectorType>
    Evaluation
    eval(const VectorType &state) const
    {
      return Evaluation(state);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE void
    apply_d2f_detaidetaj(const VectorType          &L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType       *value_result) const
    {
      (void)L;
      (void)etas;
      (void)n_grains;
      (void)value;
      (void)value_result;
    }
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class GreenquistGrandPotentialOperator
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        GreenquistGrandPotentialOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T =
      GreenquistGrandPotentialOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    GreenquistGrandPotentialOperator(
      const MatrixFree<dim, Number, VectorizedArrayType>      &matrix_free,
      const AffineConstraints<Number>                         &constraints,
      const GreenquistFreeEnergy<VectorizedArrayType>         &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>   &data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &history,
      const bool                                               matrix_based)
      : OperatorBase<dim, Number, VectorizedArrayType, T>(matrix_free,
                                                          constraints,
                                                          0,
                                                          "sintering_gp_op",
                                                          matrix_based)
      , free_energy(free_energy)
      , data(data)
      , history(history)
    {}

    ~GreenquistGrandPotentialOperator()
    {}

    template <typename... Arg>
    static T
    create(
      const MatrixFree<dim, Number, VectorizedArrayType>      &matrix_free,
      const AffineConstraints<Number>                         &constraints,
      const GreenquistFreeEnergy<VectorizedArrayType>         &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>   &sintering_data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &solution_history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
                &advection_mechanism,
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
    do_add_data_vectors_kernel(DataOut<dim>                &data_out,
                               const BlockVectorType       &vec,
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
        possible_entries = {{{"bnds", FieldBnds, 1}, {"gb", FieldGb, 1}}};

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
              const auto val = fe_eval_all.get_value(q);

              std::array<VectorizedArrayType, n_grains> etas;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                etas[ig] = val[2 + ig];

              unsigned int counter = 0;

              if (entries_mask[FieldBnds])
                temp[counter++] = PowerHelper<n_grains, 2>::power_sum(etas);

              if (entries_mask[FieldGb])
                {
                  VectorizedArrayType etaijSum = 0.0;
                  for (unsigned int i = 0; i < n_grains; ++i)
                    for (unsigned int j = 0; j < i; ++j)
                      etaijSum += etas[i + 2] * etas[j + 2];

                  temp[counter++] = etaijSum;
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
        names.push_back("bnds");

      if (entries_mask[FieldGb])
        names.push_back("gb");

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
      return this->data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2;
    }

    virtual EquationType
    equation_type(const unsigned int component) const override
    {
      (void)component;
      return EquationType::TimeDependent;
    }

    template <int with_time_derivative = 2>
    void
    evaluate_nonlinear_residual(
      BlockVectorType       &dst,
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const
    {
      MyScope scope(this->timer,
                    "sintering_gp_op::nonlinear_residual",
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
    &GreenquistGrandPotentialOperator::                           \
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
      AssertDimension(n_comp, n_grains + 2);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      const auto &free_energy = this->free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &order       = this->data.time_data.get_order();

      const auto old_solutions = this->history.get_old_solutions();

      TimeIntegration::TimeCellIntegrator time_eval(
        phi,
        this->data.time_data.get_weights().begin(),
        this->data.time_data.get_weights().end());

      for (unsigned int i = 0; i < order; ++i)
        {
          time_eval[i].reinit(cell);
          time_eval[i].read_dof_values_plain(*old_solutions[i]);
          time_eval[i].evaluate(EvaluationFlags::EvaluationFlags::values);
        }

      VectorizedArrayType dummy_one(1.0);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          typename FECellIntegratorType::value_type    value_result;
          typename FECellIntegratorType::gradient_type gradient_result;

          const auto  value        = phi.get_value(q);
          const auto  gradient     = phi.get_gradient(q);
          const auto &lin_value    = nonlinear_values[cell][q];
          const auto &lin_gradient = nonlinear_gradients[cell][q];

          const auto &lin_phi_value    = lin_value[0];
          const auto &lin_mu_value     = lin_value[1];
          const auto &lin_phi_gradient = lin_gradient[0];
          const auto &mu_lin_gradient  = lin_gradient[1];

          const auto free_energy_eval =
            free_energy.template eval<EnergySecond, n_grains>(lin_value);

          const VectorizedArrayType *lin_etas_value = &lin_value[2];
          const Tensor<1, dim, VectorizedArrayType> *lin_etas_gradient =
            nullptr;

          if (SinteringOperatorData<dim, VectorizedArrayType>::
                use_tensorial_mobility)
            lin_etas_gradient = &lin_gradient[2];

          // Evaluate some time derivatives
          VectorizedArrayType dphi_dt, dmu_dt;

          TimeIntegration::compute_time_derivative(
            dphi_dt, lin_phi_value, time_eval, 0, q);

          TimeIntegration::compute_time_derivative(
            dmu_dt, lin_mu_value, time_eval, 1, q);

          // Evaluate phi row
          value_result[0] = value[0] * weight;
          value_result[0] +=
            free_energy_eval.get_Lphi() *
            (free_energy_eval.d2hb() * free_energy_eval.omega_b() +

             free_energy_eval.d2hm() * free_energy_eval.omega_m() +
             free_energy_eval.dhm() *
               free_energy_eval.domega_m_dphasei(lin_phi_value) +

             free_energy_eval.dhm() *
               free_energy_eval.domega_m_dphasei(lin_phi_value) +
             free_energy_eval.hm() *
               free_energy_eval.d2omega_m_dphasei2(lin_phi_value) +


             free_energy_eval.d2epsilon_dphi2() * free_energy_eval.zeta() +
             free_energy_eval.depsilon_dphi() *
               free_energy_eval.dzeta_dphasei(lin_phi_value) +

             free_energy_eval.depsilon_dphi() *
               free_energy_eval.dzeta_dphasei(lin_phi_value) +
             free_energy_eval.epsilon() *
               free_energy_eval.d2zeta_dphasei2(lin_phi_value)) *
            value[0];

          value_result[0] +=
            free_energy_eval.get_Lphi() *
            (free_energy_eval.dhb() * free_energy_eval.domega_b_dmu() +
             free_energy_eval.dhm() * free_energy_eval.domega_m_dmu() +
             free_energy_eval.hm() *
               free_energy_eval.d2omega_m_dmudphasei(lin_phi_value)) *
            value[1];

          gradient_result[0] = free_energy_eval.get_Lphi() *
                                 free_energy_eval.dkappa_dphi() *
                                 lin_phi_gradient * value[0] +
                               free_energy_eval.get_Lphi() *
                                 free_energy_eval.kappa() * gradient[0];

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[0] +=
                free_energy_eval.get_Lphi() *
                (free_energy_eval.dhm() *
                   free_energy_eval.domega_m_dphasei(lin_value[ig + 2]) +

                 free_energy_eval.hm() *
                   free_energy_eval.d2omega_m_dphaseidphasej(
                     lin_phi_value, lin_value[ig + 2]) +

                 free_energy_eval.depsilon_dphi() *
                   free_energy_eval.dzeta_dphasei(lin_value[ig + 2]) +

                 free_energy_eval.epsilon() *
                   free_energy_eval.d2zeta_dphaseidphasej(lin_phi_value,
                                                          lin_value[ig + 2])) *
                value[ig + 2];
            }

          // Evaluate mu row
          value_result[1] = (value[1] * weight) * free_energy_eval.chi() +
                            dmu_dt * free_energy_eval.dchi_dphi() * value[0];

          value_result[1] -=
            (free_energy_eval.dhb() * free_energy_eval.domega_b_dmu() +
             free_energy_eval.dhm() * free_energy_eval.domega_m_dmu() +
             free_energy_eval.hm() *
               free_energy_eval.d2omega_m_dmudphasei(lin_phi_value)) *
            value[0] * weight;

          value_result[1] -=
            (free_energy_eval.d2hb() * free_energy_eval.domega_b_dmu() +
             free_energy_eval.d2hm() * free_energy_eval.domega_m_dmu() +
             free_energy_eval.dhm() *
               free_energy_eval.d2omega_m_dmudphasei(lin_phi_value) +
             free_energy_eval.dhm() *
               free_energy_eval.d2omega_m_dmudphasei(lin_phi_value) +
             free_energy_eval.hm() *
               free_energy_eval.d3omega_m_dmudphasei2(lin_phi_value)) *
            value[0] * dphi_dt;

          value_result[1] -=
            (free_energy_eval.dhb() * free_energy_eval.d2omega_b_dmu2() +
             free_energy_eval.dhm() * free_energy_eval.d2omega_b_dmu2()) *
            value[1] * dphi_dt;

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[1] -=
                (free_energy_eval.dhm() *
                   free_energy_eval.d2omega_m_dmudphasei(lin_value[ig + 2]) +
                 free_energy_eval.hm() *
                   free_energy_eval.d3omega_m_dmudphaseidphasej(
                     lin_phi_value, lin_value[ig + 2])) *
                value[ig + 2] * dphi_dt;
            }

          gradient_result[1] =
            (mobility.M_vol(dummy_one) +
             mobility.M_surf(lin_phi_value, lin_phi_gradient) +
             mobility.M_gb(lin_etas_value, n_grains, lin_etas_gradient)) *
            mu_lin_gradient * free_energy_eval.dchi_dphi() * value[0];

          gradient_result[1] +=
            (mobility.M_vol(dummy_one) +
             mobility.M_surf(lin_phi_value, lin_phi_gradient) +
             mobility.M_gb(lin_etas_value, n_grains, lin_etas_gradient)) *
            free_energy_eval.chi() * gradient[1];

          gradient_result[1] +=
            free_energy_eval.chi() *
            (mobility.dM_surf_dc(lin_phi_value, lin_phi_gradient) *
               mu_lin_gradient * value[0] +
             mobility.dM_dgrad_c(lin_phi_value,
                                 lin_phi_gradient,
                                 mu_lin_gradient) *
               gradient[0]);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              gradient_result[1] +=
                free_energy_eval.chi() * (mobility.dM_detai(lin_phi_value,
                                                            lin_etas_value,
                                                            n_grains,
                                                            lin_phi_gradient,
                                                            lin_etas_gradient,
                                                            ig) *
                                          mu_lin_gradient * value[ig + 2]);
            }

          // Evaluate etas[ig] row
          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[ig + 2] = value[ig + 2] * weight;

              value_result[ig + 2] +=
                free_energy_eval.get_L() *
                (free_energy_eval.dhm() *
                   free_energy_eval.domega_m_dphasei(lin_value[ig + 2]) +
                 free_energy_eval.hm() *
                   free_energy_eval.d2omega_m_dphaseidphasej(lin_value[ig + 2],
                                                             lin_phi_value) +
                 free_energy_eval.depsilon_dphi() *
                   free_energy_eval.dzeta_dphasei(lin_value[ig + 2]) +
                 free_energy_eval.epsilon() *
                   free_energy_eval.d2zeta_dphaseidphasej(lin_value[ig + 2],
                                                          lin_phi_value)) *
                value[0];

              value_result[ig + 2] +=
                free_energy_eval.get_L() *
                (free_energy_eval.hm() *
                 free_energy_eval.d2omega_m_dmudphasei(lin_value[ig + 2])) *
                value[1];

              value_result[ig + 2] +=
                free_energy_eval.get_L() *
                (free_energy_eval.hm() *
                   free_energy_eval.d2omega_m_dphasei2(lin_value[ig + 2]) +
                 free_energy_eval.epsilon() *
                   free_energy_eval.d2zeta_dphasei2(lin_value[ig + 2])) *
                value[ig + 2];

              for (unsigned int jg = 0; jg < n_grains; ++jg)
                if (jg != ig)
                  value_result[ig + 2] +=
                    free_energy_eval.get_L() *
                    (free_energy_eval.hm() *
                       free_energy_eval.d2omega_m_dphaseidphasej(
                         lin_value[ig + 2], lin_value[jg + 2]) +
                     free_energy_eval.epsilon() *
                       free_energy_eval.d2zeta_dphaseidphasej(
                         lin_value[ig + 2], lin_value[jg + 2])) *
                    value[jg + 2];

              gradient_result[ig + 2] = free_energy_eval.get_L() *
                                        free_energy_eval.kappa() *
                                        gradient[ig + 2];
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
                                                             &advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType                             &phi,
      typename FECellIntegratorType::value_type              &value_result,
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
                                                             &advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType                             &phi,
      typename FECellIntegratorType::value_type              &value_result,
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
                                                             &advection_data,
      const SinteringNonLinearData<dim, VectorizedArrayType> &nonlinear_data,
      const FECellIntegratorType                             &phi,
      typename FECellIntegratorType::value_type              &value_result,
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
          if (qty == "void_vol")
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
                    eta_ij_sum += value[i + 2] * value[j + 2];

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

                    return i < data.n_grains() ? value[i + 2] : 0.;
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

    const GreenquistFreeEnergy<VectorizedArrayType> &
    get_free_energy() const
    {
      return free_energy;
    }

  private:
    template <int n_comp, int n_grains, int with_time_derivative>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      AssertDimension(n_comp, n_grains + 2);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> eval(
        matrix_free, this->dof_index);

      TimeIntegration::TimeCellIntegrator time_eval(
        eval,
        this->data.time_data.get_weights().begin(),
        this->data.time_data.get_weights().end());

      const auto &mobility = this->data.get_mobility();
      const auto &order    = this->data.time_data.get_order();

      const auto old_solutions = this->history.get_old_solutions();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          eval.reinit(cell);
          eval.gather_evaluate(src,
                               EvaluationFlags::EvaluationFlags::values |
                                 EvaluationFlags::EvaluationFlags::gradients);

          if (with_time_derivative)
            for (unsigned int i = 0; i < order; ++i)
              {
                time_eval[i].reinit(cell);
                time_eval[i].read_dof_values_plain(*old_solutions[i]);
                time_eval[i].evaluate(EvaluationFlags::EvaluationFlags::values);
              }

          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            {
              const auto val  = eval.get_value(q);
              const auto grad = eval.get_gradient(q);

              auto &phi      = val[0];
              auto &phi_grad = grad[0];
              auto &mu_grad  = grad[1];

              typename FECellIntegrator<dim,
                                        n_comp,
                                        Number,
                                        VectorizedArrayType>::value_type
                value_result,
                time_derivatives;
              typename FECellIntegrator<dim,
                                        n_comp,
                                        Number,
                                        VectorizedArrayType>::gradient_type
                gradient_result;

              std::array<VectorizedArrayType, n_grains>     etas;
              std::array<VectorizedArrayType, n_grains + 1> phases;
              std::array<Tensor<1, dim, VectorizedArrayType>, n_grains>
                etas_grad;

              phases[0] = val[0];
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]       = val[2 + ig];
                  etas_grad[ig]  = grad[2 + ig];
                  phases[ig + 1] = val[2 + ig];
                }

              const auto free_energy_eval =
                free_energy.template eval<EnergyAll, n_grains>(val);

              // Evaluate time derivatives of all quantities
              if (with_time_derivative)
                for (unsigned int ic = 0; ic < n_comp; ++ic)
                  TimeIntegration::compute_time_derivative(
                    value_result[ic], val, time_eval, ic, q);

              // Copy dphi_dt for later use
              auto dphi_dt = value_result[0];

              // Modify the time derivative of mu
              value_result[1] *= free_energy_eval.chi();

              // Evaluate phi row
              value_result[0] +=
                free_energy_eval.get_Lphi() *
                (free_energy_eval.dhb() * free_energy_eval.omega_b() +
                 free_energy_eval.dhm() * free_energy_eval.omega_m() +
                 free_energy_eval.hm() *
                   free_energy_eval.domega_m_dphasei(phi) +
                 free_energy_eval.depsilon_dphi() * free_energy_eval.zeta() +
                 free_energy_eval.epsilon() *
                   free_energy_eval.dzeta_dphasei(phi));

              gradient_result[0] = free_energy_eval.get_Lphi() *
                                   free_energy_eval.kappa() * grad[0];

              // Evaluate etas[ig] row
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[ig + 2] +=
                    free_energy_eval.get_L() *
                    (free_energy_eval.hm() *
                       free_energy_eval.domega_m_dphasei(etas[ig]) +
                     free_energy_eval.epsilon() *
                       free_energy_eval.dzeta_dphasei(etas[ig]));

                  gradient_result[ig + 2] = free_energy_eval.get_L() *
                                            free_energy_eval.kappa() *
                                            etas_grad[ig];
                }

              // Evaluate chemical potential row
              dphi_dt *=
                free_energy_eval.dhb() * free_energy_eval.domega_b_dmu() +
                free_energy_eval.dhm() * free_energy_eval.domega_m_dmu() +
                free_energy_eval.hm() *
                  free_energy_eval.d2omega_m_dmudphasei(phi);
              dphi_dt *= -1.;

              value_result[1] += dphi_dt;

              VectorizedArrayType dummy_one(1.0);

              gradient_result[1] =
                free_energy_eval.chi() *
                (mobility.M_vol(dummy_one) + mobility.M_surf(phi, phi_grad) +
                 mobility.M_gb(etas, n_grains, etas_grad)) *
                mu_grad;

              eval.submit_value(value_result, q);
              eval.submit_gradient(gradient_result, q);
            }
          eval.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                   EvaluationFlags::EvaluationFlags::gradients,
                                 dst);
        }
    }

  protected:
    const GreenquistFreeEnergy<VectorizedArrayType>          free_energy;
    const SinteringOperatorData<dim, VectorizedArrayType>   &data;
    const TimeIntegration::SolutionHistory<BlockVectorType> &history;
  };
} // namespace Sintering
