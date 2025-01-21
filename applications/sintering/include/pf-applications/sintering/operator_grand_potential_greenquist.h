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
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detai2(const VectorizedArrayType &etai) const
      {
        return VectorizedArrayType(0.);
      }

      DEAL_II_ALWAYS_INLINE VectorizedArrayType
      d2f_detaidetaj(const VectorizedArrayType &etai,
                     const VectorizedArrayType &etaj) const
      {
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
    apply_d2f_detaidetaj(const VectorType &         L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType *      value_result) const
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
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const GreenquistFreeEnergy<VectorizedArrayType> &        free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType> &  data,
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
      , time_integrator(data.time_data, history)
    {}

    ~GreenquistGrandPotentialOperator()
    {}

    template <typename... Arg>
    static T
    create(
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const GreenquistFreeEnergy<VectorizedArrayType> &        free_energy,
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

  protected:
    const GreenquistFreeEnergy<VectorizedArrayType>          free_energy;
    const SinteringOperatorData<dim, VectorizedArrayType> &  data;
    const TimeIntegration::SolutionHistory<BlockVectorType> &history;
    const TimeIntegration::BDFIntegrator<dim, Number, VectorizedArrayType>
      time_integrator;
  };
} // namespace Sintering
