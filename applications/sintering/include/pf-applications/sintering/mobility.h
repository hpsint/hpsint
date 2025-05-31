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

#include <deal.II/base/tensor.h>

#include <pf-applications/numerics/functions.h>

#include <pf-applications/sintering/tools.h>

namespace Sintering
{
  using namespace dealii;
  using namespace hpsint;

  struct MobilityCoefficients
  {
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;
    double L;
  };

  class MobilityProvider
  {
  public:
    virtual MobilityCoefficients
    calculate(const double t) const = 0;
  };

  class ProviderAbstract : public MobilityProvider
  {
  public:
    ProviderAbstract(const double Mvol,
                     const double Mvap,
                     const double Msurf,
                     const double Mgb,
                     const double L)
      : Mvol(Mvol)
      , Mvap(Mvap)
      , Msurf(Msurf)
      , Mgb(Mgb)
      , L(L)
    {}

    MobilityCoefficients
    calculate(const double t) const override
    {
      (void)t;

      MobilityCoefficients mobilities{Mvol, Mvap, Msurf, Mgb, L};

      return mobilities;
    }

  protected:
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;
    double L;
  };

  class ProviderRealistic : public MobilityProvider
  {
  public:
    enum class ArrheniusUnit
    {
      Boltzmann,
      Gas
    };

    ProviderRealistic(
      const double                        omega,
      const double                        D_vol0,
      const double                        D_vap0,
      const double                        D_surf0,
      const double                        D_gb0,
      const double                        Q_vol,
      const double                        Q_vap,
      const double                        Q_surf,
      const double                        Q_gb,
      const double                        D_gb_mob0,
      const double                        Q_gb_mob,
      const double                        interface_width,
      const double                        time_scale,
      const double                        length_scale,
      const double                        energy_scale,
      std::shared_ptr<Function1D<double>> temperature,
      const ArrheniusUnit arrhenius_unit = ArrheniusUnit::Boltzmann)
      : omega(omega)
      , D_vol0(D_vol0)
      , D_vap0(D_vap0)
      , D_surf0(D_surf0)
      , D_gb0(D_gb0)
      , Q_vol(Q_vol)
      , Q_vap(Q_vap)
      , Q_surf(Q_surf)
      , Q_gb(Q_gb)
      , D_gb_mob0(D_gb_mob0)
      , Q_gb_mob(Q_gb_mob)
      , interface_width(interface_width)
      , time_scale(time_scale)
      , length_scale(length_scale)
      , energy_scale(energy_scale)
      , omega_dmls(omega / std::pow(length_scale, 3))
      , arrhenius_unit(arrhenius_unit)
      , arrhenius_factor(arrhenius_unit == ArrheniusUnit::Boltzmann ? kb : R)
      , temperature(temperature)
    {}

    MobilityCoefficients
    calculate(const double time) const override
    {
      (void)time;

      MobilityCoefficients mobilities;

      const double T = temperature->value(time);

      mobilities.Mvol  = calc_diffusion_coefficient(D_vol0, Q_vol, T);
      mobilities.Mvap  = calc_diffusion_coefficient(D_vap0, Q_vap, T);
      mobilities.Msurf = calc_diffusion_coefficient(D_surf0, Q_surf, T);
      mobilities.Mgb   = calc_diffusion_coefficient(D_gb0, Q_gb, T);
      mobilities.L     = calc_gb_mobility_coefficient(D_gb_mob0, Q_gb_mob, T);

      return mobilities;
    }

  private:
    double
    calc_diffusion_coefficient(const double D0,
                               const double Q,
                               const double T) const
    {
      // Non-dimensionalized Diffusivity prefactor
      const double D0_dmls = D0 * time_scale / (length_scale * length_scale);
      const double D_dmls  = D0_dmls * std::exp(-Q / (arrhenius_factor * T));
      const double M =
        D_dmls * omega_dmls / (arrhenius_factor * T) * energy_scale;

      return M;
    }

    double
    calc_gb_mobility_coefficient(const double D0,
                                 const double Q,
                                 const double T) const
    {
      // Convert to lengthscale^4/(eV*timescale);
      const double D0_dmls =
        D0 * time_scale * energy_scale / std::pow(length_scale, 4);
      const double D_dmls = D0_dmls * std::exp(-Q / (arrhenius_factor * T));
      const double L      = 4.0 / 3.0 * D_dmls / interface_width;

      return L;
    }

    // Some constants
    const double kb = 8.617343e-5; // Boltzmann constant in eV/K
    const double R  = 8.314;       // Gas constant J / (mol K)

    // atomic volume
    const double omega;

    // prefactors
    const double D_vol0;
    const double D_vap0;
    const double D_surf0;
    const double D_gb0;

    // activation energies
    const double Q_vol;
    const double Q_vap;
    const double Q_surf;
    const double Q_gb;

    // GB mobility coefficients
    const double D_gb_mob0;
    const double Q_gb_mob;

    // Interface width
    const double interface_width;

    // Scales
    const double time_scale;
    const double length_scale;
    const double energy_scale;

    // Dimensionless omega
    const double omega_dmls;

    // Arrhenius units
    const ArrheniusUnit arrhenius_unit;
    const double        arrhenius_factor;

    // Temperature function
    std::shared_ptr<Function1D<double>> temperature;
  };

  class Mobility
  {
  public:
    Mobility(std::shared_ptr<MobilityProvider> provider)
      : provider(provider)
    {
      update(0.);
    }

    void
    update(const double time)
    {
      const auto mobilities = provider->calculate(time);

      Mvol  = mobilities.Mvol;
      Mvap  = mobilities.Mvap;
      Msurf = mobilities.Msurf;
      Mgb   = mobilities.Mgb;
      L     = mobilities.L;
    }

  protected:
    std::shared_ptr<MobilityProvider> provider;

    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;
    double L;
  };

  template <int dim, typename VectorizedArrayType>
  class MobilityScalar : public Mobility
  {
  public:
    /**
     * Computes scalar mobility as a sum of 4 components:
     *   - volumetric (bulk) diffusion,
     *   - vaporization diffusion,
     *   - surface diffusion,
     *   - grain-boundary diffusion.
     *
     * Three options are available for the scalar surface diffusion mobility:
     * @f[
     *   M_{surf1} = c (1-c)
     * @f]
     * and
     * @f[
     *   M_{surf2} = c^2 (1-c^2),
     * @f]
     * and
     * @f[
     *   M_{surf3} = 4 c^2 (1-c)^2,
     * @f]
     * where $c$ is the concentration. The first option was implemented at the
     * beginning. However, it turned out that this term significantly worsens
     * the convergence of nonlinear Newton iterations. The residual of CH block
     * converges slowly leading to a large number of iterations within the
     * timestep and that limits the growth of $\Delta t$. Surprisingly, this
     * issue does not emerge if the tangent matrix of the system is computed
     * with finite differences. Probably, this strange behavior is provoked by
     * the following bounding of $c$ appearing earlier in the code:
     * @f[
     *   0 \leq < c \leq 1.
     * @f]
     * This restriction is essential for this mobility term since it ensures
     * that $M_{surf} \geq 0$ always. It is possible to observe in numerical
     * simulations that once this condition is violated, the solution diverges
     * quite fast, so it is hardly possible to get rid of it.
     *
     * The second option renders a significantly better behavior: the analytical
     * tangent seems to be much better in comparison to the first option.
     * Iterations of nonlinear solver converge much faster and, in general, the
     * convergence observed in the numerical experiments agrees perfectly with
     * the one obtained by using the tangent computed with finite differences.
     * Furthermore, this mobility term, disticntly from the first option, does
     * not break simulations if $c<0$: in such a situation the mobility term
     * still remains positive. This allows one to use a relaxed bounding:
     * @f[
     *   c \leq 1.
     * @f]
     *
     * The third option is not sensitive to the issues of $c$ being $<0$ or
     * $>1$, however it is less a studied function that has never appeared in
     * literature before. Due to the form of the function, it requires a scalar
     * prefactor 4 to achieve dynamics comparable with the other two variants.
     *
     * Due to these reasons, the third second option is now implemented in the
     * code. The checks for $0 \leq < c \leq 1$, previously existing, have been
     * removed from the code.
     */
    MobilityScalar(std::shared_ptr<MobilityProvider> provider)
      : Mobility(provider)
    {}



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_vol(const VectorizedArrayType &c) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      return Mvol * phi;
    }



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_vap(const VectorizedArrayType &c) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      return Mvap * (1.0 - phi);
    }



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_surf(const VectorizedArrayType                 &c,
           const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      (void)c_grad;

      return Msurf * 4.0 * c * c * (1.0 - c) * (1.0 - c);
    }



    // note: only for postprocessing
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_gb(const VectorTypeValue    &etas,
         const unsigned int        etas_size,
         const VectorTypeGradient &etas_grad) const
    {
      (void)etas_grad;

      VectorizedArrayType etaijSum = 0.0;
      for (unsigned int i = 0; i < etas_size; ++i)
        for (unsigned int j = 0; j < i; ++j)
          etaijSum += etas[i] * etas[j];
      etaijSum *= 2.0;

      return Mgb * etaijSum;
    }



    // TODO: replace by apply!?
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_dc(const VectorizedArrayType                 &c,
          const VectorTypeValue                     &etas,
          const Tensor<1, dim, VectorizedArrayType> &c_grad,
          const VectorTypeGradient                  &etas_grad) const
    {
      (void)etas;
      (void)c_grad;
      (void)etas_grad;

      const VectorizedArrayType dphidc = 30.0 * c * c * (1.0 - c) * (1.0 - c);
      const VectorizedArrayType dMdc =
        Mvol * dphidc - Mvap * dphidc +
        Msurf * 8.0 * c * (1.0 - 3.0 * c + 2.0 * c * c);

      return dMdc;
    }



    // TODO: replace by apply!?
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType                 &c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      (void)c;
      (void)c_grad;
      (void)mu_grad;

      return Tensor<2, dim, VectorizedArrayType>();
    }



    // TODO: replace by apply!?
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_detai(const VectorizedArrayType                 &c,
             const VectorTypeValue                     &etas,
             const unsigned int                         etas_size,
             const Tensor<1, dim, VectorizedArrayType> &c_grad,
             const VectorTypeGradient                  &etas_grad,
             unsigned int                               index_i) const
    {
      (void)c;
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType etajSum = 0;
      for (unsigned int j = 0; j < etas_size; ++j)
        if (j != index_i)
          etajSum += etas[j];

      const auto MetajSum = 2.0 * Mgb * etajSum;

      return MetajSum;
    }



    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
                          apply_M(const VectorizedArrayType                 &lin_c_value,
                                  const VectorTypeValue                     &lin_etas_value,
                                  const unsigned int                         n_grains,
                                  const Tensor<1, dim, VectorizedArrayType> &lin_c_gradient,
                                  const VectorTypeGradient                  &lin_etas_gradient,
                                  const Tensor<1, dim, VectorizedArrayType> &mu_gradient) const
    {
      (void)lin_c_gradient;
      (void)lin_etas_gradient;

      // warning: nested loop over grains; optimization: exploit symmetry
      // and only loop over lower-triangular matrix
      VectorizedArrayType etaijSum = 0.0;
      for (unsigned int i = 0; i < n_grains; ++i)
        for (unsigned int j = 0; j < i; ++j)
          etaijSum += lin_etas_value[i] * lin_etas_value[j];

      VectorizedArrayType phi =
        lin_c_value * lin_c_value * lin_c_value *
        (10.0 - 15.0 * lin_c_value + 6.0 * lin_c_value * lin_c_value);

      phi = std::max(VectorizedArrayType(0.0), phi);
      phi = std::min(VectorizedArrayType(1.0), phi);

      const VectorizedArrayType M = Mvol * phi + Mvap * (1.0 - phi) +
                                    Msurf * 4.0 * lin_c_value * lin_c_value *
                                      (1.0 - lin_c_value) *
                                      (1.0 - lin_c_value) +
                                    (2.0 * Mgb) * etaijSum;

      return M * mu_gradient;
    }



    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
                          apply_M_derivative(
                            const VectorizedArrayType                 *lin_value,
                            const Tensor<1, dim, VectorizedArrayType> *lin_gradient,
                            const unsigned int                         n_grains,
                            const VectorizedArrayType                 *value,
                            const Tensor<1, dim, VectorizedArrayType> *gradient) const
    {
      const auto &lin_c_value     = lin_value[0];
      const auto  lin_etas_value  = lin_value + 2;
      const auto &lin_mu_gradient = lin_gradient[1];
      const auto &c_value         = value[0];
      const auto  etas_value      = value + 2;
      const auto &mu_gradient     = gradient[1];

      // 1) for M
      // warning: nested loop over grains; optimization: exploit symmetry
      // and only loop over lower-triangular matrix
      VectorizedArrayType eta_sum    = 0.0;
      VectorizedArrayType eta_ij_sum = 0.0;
      for (unsigned int i = 0; i < n_grains; ++i)
        {
          eta_sum += lin_etas_value[i];

          for (unsigned int j = 0; j < i; ++j)
            eta_ij_sum += lin_etas_value[i] * lin_etas_value[j];
        }

      VectorizedArrayType phi =
        lin_c_value * lin_c_value * lin_c_value *
        (10.0 - 15.0 * lin_c_value + 6.0 * lin_c_value * lin_c_value);

      phi = std::max(VectorizedArrayType(0.0), phi);
      phi = std::min(VectorizedArrayType(1.0), phi);

      const VectorizedArrayType M = Mvol * phi + Mvap * (1.0 - phi) +
                                    Msurf * 4.0 * lin_c_value * lin_c_value *
                                      (1.0 - lin_c_value) *
                                      (1.0 - lin_c_value) +
                                    (2.0 * Mgb) * eta_ij_sum;

      // 2) for dM_dc
      const VectorizedArrayType dphidc = 30.0 * lin_c_value * lin_c_value *
                                         (1.0 - lin_c_value) *
                                         (1.0 - lin_c_value);
      const VectorizedArrayType dMdc =
        Mvol * dphidc - Mvap * dphidc +
        Msurf * 8.0 * lin_c_value *
          (1.0 - 3.0 * lin_c_value + 2.0 * lin_c_value * lin_c_value);

      // 3) for dM_detai
      VectorizedArrayType factor = 0.0;
      for (unsigned int i = 0; i < n_grains; ++i)
        factor += (eta_sum - lin_etas_value[i]) * etas_value[i];

      // result
      return M * mu_gradient +
             (dMdc * c_value + (2.0 * Mgb) * factor) * lin_mu_gradient;
    }



    // TODO: add apply!?
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_vol_dc(const VectorizedArrayType &c) const
    {
      const VectorizedArrayType dphidc = 30.0 * c * c * (1.0 - c) * (1.0 - c);
      const VectorizedArrayType dMdc   = Mvol * dphidc;

      return dMdc;
    }

    // TODO: add apply!?
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_surf_dc(const VectorizedArrayType                 &c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      (void)c_grad;

      const auto factor = 8.0 * c * (1.0 - 3.0 * c + 2.0 * c * c);

      const auto dMdc = diagonal_matrix<dim>(Msurf * factor);

      return dMdc;
    }


    double
    Lgb() const
    {
      return L;
    }
  };



  template <int dim, typename VectorizedArrayType>
  class MobilityTensorial : public Mobility
  {
  public:
    MobilityTensorial(std::shared_ptr<MobilityProvider> provider)
      : Mobility(provider)
    {}



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_vol(const VectorizedArrayType &c) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M = diagonal_matrix<dim>(Mvol * phi);

      return M;
    }



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_vap(const VectorizedArrayType &c) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M =
        diagonal_matrix<dim>(Mvap * (1. - phi));

      return M;
    }



    // note: only for postprocessing
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_surf(const VectorizedArrayType                 &c,
                                 const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      // Surface anisotropic part
      VectorizedArrayType fsurf = Msurf * 4.0 * (c * c) * (1. - c) * (1. - c);
      Tensor<1, dim, VectorizedArrayType> nc = unit_vector(c_grad);
      Tensor<2, dim, VectorizedArrayType> M  = projector_matrix(nc, fsurf);

      return M;
    }



    // note: only for postprocessing
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_gb(const VectorTypeValue    &etas,
                               const unsigned int        etas_size,
                               const VectorTypeGradient &etas_grad) const
    {
      Tensor<2, dim, VectorizedArrayType> M;

      // GB diffusion part
      for (unsigned int i = 0; i < etas_size; ++i)
        {
          for (unsigned int j = 0; j < etas_size; ++j)
            {
              if (i != j)
                {
                  VectorizedArrayType fgb = Mgb * (etas[i]) * (etas[j]);
                  Tensor<1, dim, VectorizedArrayType> eta_grad_diff =
                    (etas_grad[i]) - (etas_grad[j]);
                  Tensor<1, dim, VectorizedArrayType> neta =
                    unit_vector(eta_grad_diff);
                  M += projector_matrix(neta, fgb);
                }
            }
        }

      return M;
    }



    // TODO: replace by apply!?
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dc(const VectorizedArrayType                 &c,
                                const VectorTypeValue                     &etas,
                                const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                const VectorTypeGradient                  &etas_grad) const
    {
      (void)etas;
      (void)etas_grad;

      const auto c2_1minusc2 = 4.0 * c * c * (1. - c) * (1. - c);
      const auto dphidc      = 30.0 * c2_1minusc2;

      // Volumetric and vaporization parts, the same as for isotropic
      auto dMdc = diagonal_matrix<dim>((Mvol - Mvap) * dphidc);

      // Surface part
      const auto fsurf  = Msurf * c2_1minusc2;
      auto       dfsurf = Msurf * 8.0 * c * (1. - c) * (1. - 2. * c);

      dfsurf = compare_and_apply_mask<SIMDComparison::less_than>(
        fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), dfsurf);

      const auto nc = unit_vector(c_grad);
      dMdc +=
        projector_matrix(nc, dfsurf, VectorizedArrayType(projector_scale));

      return dMdc;
    }



    // TODO: replace by apply!?
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType                 &c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      auto fsurf = Msurf * 4.0 * (c * c) * ((1. - c) * (1. - c));
      auto nrm   = c_grad.norm();

      fsurf = compare_and_apply_mask<SIMDComparison::less_than>(
        fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), fsurf);
      nrm = compare_and_apply_mask<SIMDComparison::less_than>(
        nrm, VectorizedArrayType(1e-4), VectorizedArrayType(1.0), nrm);

      const auto nc = unit_vector(c_grad);
      const auto M  = projector_matrix(nc, 1. / nrm);

      auto T = diagonal_matrix<dim>(nc * mu_grad) + outer_product(nc, mu_grad);
      T *= -fsurf;

      return T * M;
    }



    // TODO: replace by apply!?
    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_detai(const VectorizedArrayType                 &c,
                                   const VectorTypeValue                     &etas,
                                   const unsigned int                         etas_size,
                                   const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                   const VectorTypeGradient                  &etas_grad,
                                   unsigned int                               index_i) const
    {
      (void)c;
      (void)c_grad;

      dealii::Tensor<2, dim, VectorizedArrayType> M;

      for (unsigned int j = 0; j < etas_size; ++j)
        {
          if (j != index_i)
            {
              const auto eta_grad_diff = etas_grad[index_i] - etas_grad[j];
              const auto neta          = unit_vector(eta_grad_diff);
              M += projector_matrix(neta, etas[j]);
            }
        }
      M *= 2. * Mgb;

      return M;
    }



    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
                          apply_M(const VectorizedArrayType                 &lin_c_value,
                                  const VectorTypeValue                     &lin_etas_value,
                                  const unsigned int                         n_grains,
                                  const Tensor<1, dim, VectorizedArrayType> &lin_c_gradient,
                                  const VectorTypeGradient                  &lin_etas_gradient,
                                  const Tensor<1, dim, VectorizedArrayType> &mu_gradient) const
    {
      VectorizedArrayType phi =
        lin_c_value * lin_c_value * lin_c_value *
        (10.0 - 15.0 * lin_c_value + 6.0 * lin_c_value * lin_c_value);

      phi = std::max(VectorizedArrayType(0.0), phi);
      phi = std::min(VectorizedArrayType(1.0), phi);

      // Volumetric and vaporization parts, the same as for isotropic
      const auto f_vol_vap = Mvol * phi + Mvap * (1.0 - phi);

      // Surface anisotropic part
      const auto fsurf = Msurf * 4.0 * (lin_c_value * lin_c_value) *
                         ((1. - lin_c_value) * (1. - lin_c_value));
      const auto nc = unit_vector(lin_c_gradient);

      const auto out = (f_vol_vap + fsurf * projector_scale) * mu_gradient -
                       nc * (fsurf * (nc * mu_gradient));

      if (n_grains <= 1)
        return out;

      // GB diffusion part
      //
      // warning: nested loop over grains; optimization: exploit symmetry
      // and only loop over lower-triangular matrix
      Tensor<1, dim, VectorizedArrayType> out_gb;
      for (unsigned int i = 0; i < n_grains; ++i)
        for (unsigned int j = 0; j < i; ++j)
          {
            const auto eta_grad_diff =
              lin_etas_gradient[i] - lin_etas_gradient[j];
            const auto filter = unit_vector_filter_2(eta_grad_diff);
            out_gb +=
              (mu_gradient -
               eta_grad_diff * (filter * (eta_grad_diff * mu_gradient))) *
              (lin_etas_value[i] * lin_etas_value[j]);
          }

      return out + out_gb * (2.0 * Mgb);
    }



    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
                          apply_M_derivative(
                            const VectorizedArrayType                 *lin_value,
                            const Tensor<1, dim, VectorizedArrayType> *lin_gradient,
                            const unsigned int                         n_grains,
                            const VectorizedArrayType                 *value,
                            const Tensor<1, dim, VectorizedArrayType> *gradient) const
    {
      const auto &lin_c_value       = lin_value[0];
      const auto  lin_etas_value    = lin_value + 2;
      const auto &lin_c_gradient    = lin_gradient[0];
      const auto &lin_mu_gradient   = lin_gradient[1];
      const auto &lin_etas_gradient = lin_gradient + 2;
      const auto &c_value           = value[0];
      const auto  etas_value        = value + 2;
      const auto &c_gradient        = gradient[0];
      const auto &mu_gradient       = gradient[1];

      Tensor<1, dim, VectorizedArrayType> out, out_gb;

      const auto nc = unit_vector(lin_c_gradient);

      // 1) for M
      {
        VectorizedArrayType phi =
          lin_c_value * lin_c_value * lin_c_value *
          (10.0 - 15.0 * lin_c_value + 6.0 * lin_c_value * lin_c_value);

        phi = std::max(VectorizedArrayType(0.0), phi);
        phi = std::min(VectorizedArrayType(1.0), phi);

        // Volumetric and vaporization parts, the same as for isotropic
        const auto f_vol_vap = Mvol * phi + Mvap * (1.0 - phi);

        // Surface anisotropic part
        const auto fsurf = Msurf * 4.0 * (lin_c_value * lin_c_value) *
                           ((1. - lin_c_value) * (1. - lin_c_value));

        out =
          (f_vol_vap + fsurf) * mu_gradient - nc * (fsurf * (nc * mu_gradient));
      }

      // 2) for dM_dc
      {
        const auto c2_1minusc2 = 4.0 * lin_c_value * lin_c_value *
                                 (1. - lin_c_value) * (1. - lin_c_value);
        const auto dphidc = 30.0 * c2_1minusc2;

        // Volumetric and vaporization parts, the same as for isotropic
        const auto f_vol_vap = (Mvol - Mvap) * dphidc;

        // Surface part
        const auto fsurf  = Msurf * c2_1minusc2;
        auto       dfsurf = Msurf * 8.0 * lin_c_value * (1. - lin_c_value) *
                      (1. - 2. * lin_c_value);

        dfsurf = compare_and_apply_mask<SIMDComparison::less_than>(
          fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), dfsurf);

        out += ((f_vol_vap + dfsurf * projector_scale) * lin_mu_gradient -
                nc * (dfsurf * (nc * lin_mu_gradient))) *
               c_value;
      }

      // 3) for dM_dgrad_c
      {
        auto fsurf = Msurf * 4.0 * (lin_c_value * lin_c_value) *
                     ((1. - lin_c_value) * (1. - lin_c_value));
        auto nrm = lin_c_gradient.norm();

        fsurf = compare_and_apply_mask<SIMDComparison::less_than>(
          fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), fsurf);
        nrm = compare_and_apply_mask<SIMDComparison::less_than>(
          nrm, VectorizedArrayType(1e-4), VectorizedArrayType(1.0), nrm);

        const auto temp = c_gradient - nc * (nc * c_gradient);
        out += (-fsurf / nrm) *
               ((nc * lin_mu_gradient) * temp + nc * (lin_mu_gradient * temp));
      }

      if (n_grains <= 1)
        return out;

      // 4) for M (gb part) and for dM_detai
      {
        // warning: nested loop over grains; optimization: exploit symmetry
        // and only loop over lower-triangular matrix
        for (unsigned int i = 0; i < n_grains; ++i)
          for (unsigned int j = 0; j < i; ++j)
            {
              const auto eta_grad_diff =
                lin_etas_gradient[i] - lin_etas_gradient[j];
              const auto filter = unit_vector_filter_2(eta_grad_diff);

              out_gb +=
                (mu_gradient -
                 eta_grad_diff * (filter * (eta_grad_diff * mu_gradient))) *
                (lin_etas_value[i] * lin_etas_value[j]);

              out_gb +=
                (lin_mu_gradient -
                 (filter * eta_grad_diff) * (eta_grad_diff * lin_mu_gradient)) *
                (lin_etas_value[j] * etas_value[i] +
                 lin_etas_value[i] * etas_value[j]);
            }
      }

      return out + out_gb * (2.0 * Mgb);
    }



    // TODO: add apply!?
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_vol_dc(const VectorizedArrayType &c) const
    {
      const auto dphidc = 30.0 * c * c * (1.0 - c) * (1.0 - c);

      const auto dMdc = diagonal_matrix<dim>(Mvol * dphidc);

      return dMdc;
    }

    // TODO: add apply!?
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_surf_dc(const VectorizedArrayType                 &c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      const auto c2_1minusc2 = 4.0 * c * c * (1. - c) * (1. - c);

      // Surface part
      const auto fsurf  = Msurf * c2_1minusc2;
      auto       dfsurf = Msurf * 8.0 * c * (1. - c) * (1. - 2. * c);

      dfsurf = compare_and_apply_mask<SIMDComparison::less_than>(
        fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), dfsurf);

      const auto nc = unit_vector(c_grad);
      const auto dMdc =
        projector_matrix(nc, dfsurf, VectorizedArrayType(projector_scale));

      return dMdc;
    }


    double
    Lgb() const
    {
      return L;
    }

  private:
    const typename VectorizedArrayType::value_type projector_scale = 1.001;
  };

} // namespace Sintering
