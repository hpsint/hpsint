#pragma once

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <pf-applications/numerics/functions.h>

#include <pf-applications/sintering/tools.h>

#include <pf-applications/dofs/dof_tools.h>
#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/matrix_free/tools.h>
#include <pf-applications/time_integration/solution_history.h>
#include <pf-applications/time_integration/time_integrators.h>

#include <fstream>

template <typename T>
using n_grains_t = decltype(std::declval<T const>().n_grains());

template <typename T>
using n_grains_to_n_components_t =
  decltype(std::declval<T const>().n_grains_to_n_components(
    std::declval<const unsigned int>()));

template <typename T>
constexpr bool has_n_grains_method =
  dealii::internal::is_supported_operation<n_grains_t, T>
    &&dealii::internal::is_supported_operation<n_grains_to_n_components_t, T>;

// clang-format off
/**
 * Macro that converts a runtime number (n_components() or n_grains())
 * to constant expressions that can be used for templating and calles
 * the provided function with the two parameters: 1) number of
 * components and 2) number of grains (if it makes sence; else -1).
 *
 * The relation between number of components and number of grains
 * is encrypted in the method T::n_grains_to_n_components().
 * 
 * The function can be used the following way:
 * ```
 * #define OPERATION(c, d) std::cout << c << " " << d << std::endl;
 * EXPAND_OPERATIONS(OPERATION);
 * #undef OPERATION
 * ```
 */
#define EXPAND_OPERATIONS(OPERATION)                                                                                  \
  if constexpr(has_n_grains_method<T>)                                                                                \
    {                                                                                                                 \
      constexpr int max_grains = MAX_SINTERING_GRAINS;                                                                \
      const unsigned int n_grains = static_cast<const T&>(*this).n_grains();                                          \
      AssertIndexRange(n_grains, max_grains + 1);                                                                     \
      switch (n_grains)                                                                                               \
        {                                                                                                             \
          case  2: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  2)), std::min(max_grains,  2)); break; \
          case  3: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  3)), std::min(max_grains,  3)); break; \
          case  4: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  4)), std::min(max_grains,  4)); break; \
          case  5: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  5)), std::min(max_grains,  5)); break; \
          case  6: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  6)), std::min(max_grains,  6)); break; \
          case  7: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  7)), std::min(max_grains,  7)); break; \
          case  8: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  8)), std::min(max_grains,  8)); break; \
          case  9: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  9)), std::min(max_grains,  9)); break; \
          case 10: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 10)), std::min(max_grains, 10)); break; \
          case 11: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 11)), std::min(max_grains, 11)); break; \
          case 12: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 12)), std::min(max_grains, 12)); break; \
          case 13: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 13)), std::min(max_grains, 13)); break; \
          case 14: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 14)), std::min(max_grains, 14)); break; \
          default:                                                                                                    \
            AssertThrow(false, ExcNotImplemented());                                                                  \
        }                                                                                                             \
    }                                                                                                                 \
  else                                                                                                                \
    {                                                                                                                 \
      constexpr int max_components = MAX_SINTERING_GRAINS + 2;                                                        \
      AssertIndexRange(this->n_components(), max_components + 1);                                                     \
      switch (this->n_components())                                                                                   \
        {                                                                                                             \
          case  1: OPERATION(std::min(max_components,  1), -1); break;                                                \
          case  2: OPERATION(std::min(max_components,  2), -1); break;                                                \
          case  3: OPERATION(std::min(max_components,  3), -1); break;                                                \
          case  4: OPERATION(std::min(max_components,  4), -1); break;                                                \
          case  5: OPERATION(std::min(max_components,  5), -1); break;                                                \
          case  6: OPERATION(std::min(max_components,  6), -1); break;                                                \
          case  7: OPERATION(std::min(max_components,  7), -1); break;                                                \
          case  8: OPERATION(std::min(max_components,  8), -1); break;                                                \
          case  9: OPERATION(std::min(max_components,  9), -1); break;                                                \
          case 10: OPERATION(std::min(max_components, 10), -1); break;                                                \
          case 11: OPERATION(std::min(max_components, 11), -1); break;                                                \
          case 12: OPERATION(std::min(max_components, 12), -1); break;                                                \
          case 13: OPERATION(std::min(max_components, 13), -1); break;                                                \
          case 14: OPERATION(std::min(max_components, 14), -1); break;                                                \
          default:                                                                                                    \
            AssertThrow(false, ExcNotImplemented());                                                                  \
        }                                                                                                             \
  }
// clang-format on

namespace Sintering
{
  using namespace dealii;

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
    calculate(const double t) const
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
    calculate(const double time) const
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

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M(const VectorizedArrayType &                c,
      const VectorTypeValue &                    etas,
      const unsigned int                         etas_size,
      const Tensor<1, dim, VectorizedArrayType> &c_grad,
      const VectorTypeGradient &                 etas_grad) const
    {
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType etaijSum = 0.0;
      for (unsigned int i = 0; i < etas_size; ++i)
        for (unsigned int j = 0; j < i; ++j)
          etaijSum += etas[i] * etas[j];
      etaijSum *= 2.0;

      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      const VectorizedArrayType M =
        Mvol * phi + Mvap * (1.0 - phi) +
        Msurf * 4.0 * c * c * (1.0 - c) * (1.0 - c) + Mgb * etaijSum;

      return M;
    }

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

    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_surf(const VectorizedArrayType &                c,
           const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      (void)c_grad;

      return Msurf * 4.0 * c * c * (1.0 - c) * (1.0 - c);
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M_gb(const VectorTypeValue &   etas,
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

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_dc(const VectorizedArrayType &                c,
          const VectorTypeValue &                    etas,
          const Tensor<1, dim, VectorizedArrayType> &c_grad,
          const VectorTypeGradient &                 etas_grad) const
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

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      (void)c;
      (void)c_grad;
      (void)mu_grad;

      return Tensor<2, dim, VectorizedArrayType>();
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_detai(const VectorizedArrayType &                c,
             const VectorTypeValue &                    etas,
             const unsigned int                         etas_size,
             const Tensor<1, dim, VectorizedArrayType> &c_grad,
             const VectorTypeGradient &                 etas_grad,
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

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M(const VectorizedArrayType &                c,
                            const VectorTypeValue &                    etas,
                            const unsigned int                         etas_size,
                            const Tensor<1, dim, VectorizedArrayType> &c_grad,
                            const VectorTypeGradient &                 etas_grad) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M =
        diagonal_matrix(Mvol * phi + Mvap * (1.0 - phi));

      // Surface anisotropic part
      VectorizedArrayType fsurf = Msurf * (c * c) * ((1. - c) * (1. - c));
      Tensor<1, dim, VectorizedArrayType> nc = unit_vector(c_grad);
      M += projector_matrix(nc, fsurf);

      // GB diffusion part
      for (unsigned int i = 0; i < etas_size; i++)
        {
          for (unsigned int j = 0; j < etas_size; j++)
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

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_vol(const VectorizedArrayType &c) const
    {
      VectorizedArrayType phi = c * c * c * (10.0 - 15.0 * c + 6.0 * c * c);

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M = diagonal_matrix(Mvol * phi);

      return M;
    }

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
        diagonal_matrix(Mvap * (1. - phi));

      return M;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_surf(const VectorizedArrayType &                c,
                                 const Tensor<1, dim, VectorizedArrayType> &c_grad) const
    {
      // Surface anisotropic part
      VectorizedArrayType fsurf = Msurf * (c * c) * (1. - c) * (1. - c);
      Tensor<1, dim, VectorizedArrayType> nc = unit_vector(c_grad);
      Tensor<2, dim, VectorizedArrayType> M  = projector_matrix(nc, fsurf);

      return M;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M_gb(const VectorTypeValue &   etas,
                               const unsigned int        etas_size,
                               const VectorTypeGradient &etas_grad) const
    {
      Tensor<2, dim, VectorizedArrayType> M;

      // GB diffusion part
      for (unsigned int i = 0; i < etas_size; i++)
        {
          for (unsigned int j = 0; j < etas_size; j++)
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

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dc(const VectorizedArrayType &                c,
                                const VectorTypeValue &                    etas,
                                const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                const VectorTypeGradient &                 etas_grad) const
    {
      (void)etas;
      (void)etas_grad;

      VectorizedArrayType c2_1minusc2 = c * c * (1. - c) * (1. - c);

      VectorizedArrayType dphidc = 30.0 * c2_1minusc2;

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> dMdc =
        diagonal_matrix((Mvol - Mvap) * dphidc);

      // Surface part
      VectorizedArrayType fsurf  = Msurf * c2_1minusc2;
      VectorizedArrayType dfsurf = Msurf * 2. * c * (1. - c) * (1. - 2. * c);

      dfsurf = compare_and_apply_mask<SIMDComparison::less_than>(
        fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), dfsurf);

      Tensor<1, dim, VectorizedArrayType> nc = unit_vector(c_grad);
      dMdc += projector_matrix(nc, dfsurf);

      return dMdc;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      VectorizedArrayType fsurf = Msurf * (c * c) * ((1. - c) * (1. - c));
      VectorizedArrayType nrm   = c_grad.norm();

      fsurf = compare_and_apply_mask<SIMDComparison::less_than>(
        fsurf, VectorizedArrayType(1e-6), VectorizedArrayType(0.0), fsurf);
      nrm = compare_and_apply_mask<SIMDComparison::less_than>(
        nrm, VectorizedArrayType(1e-4), VectorizedArrayType(1.0), nrm);

      Tensor<1, dim, VectorizedArrayType> nc = unit_vector(c_grad);
      Tensor<2, dim, VectorizedArrayType> M  = projector_matrix(nc, 1. / nrm);

      Tensor<2, dim, VectorizedArrayType> T =
        diagonal_matrix(mu_grad * nc) + outer_product(nc, mu_grad);
      T *= -fsurf;

      return T * M;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_detai(const VectorizedArrayType &                c,
                                   const VectorTypeValue &                    etas,
                                   const unsigned int                         etas_size,
                                   const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                   const VectorTypeGradient &                 etas_grad,
                                   unsigned int                               index_i) const
    {
      (void)c;
      (void)c_grad;

      dealii::Tensor<2, dim, VectorizedArrayType> M;

      for (unsigned int j = 0; j < etas_size; j++)
        {
          if (j != index_i)
            {
              Tensor<1, dim, VectorizedArrayType> eta_grad_diff =
                (etas_grad[index_i]) - (etas_grad[j]);
              Tensor<1, dim, VectorizedArrayType> neta =
                unit_vector(eta_grad_diff);
              M += projector_matrix(neta, etas[j]);
            }
        }
      M *= 2. * Mgb;

      return M;
    }

    double
    Lgb() const
    {
      return L;
    }
  };

  template <unsigned int n, std::size_t p>
  class PowerHelper
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      T initial = 0.0;

      for (unsigned int i = 0; i < n; ++i)
        initial += Utilities::fixed_power<p>(etas[i]);

      return initial;
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, n> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::vector<T> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }
  };

  template <>
  class PowerHelper<2, 2>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }
  };

  template <>
  class PowerHelper<2, 3>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }
  };

  template <class T>
  class SizeHelper;

  template <class T, std::size_t n>
  class SizeHelper<std::array<T, n>>
  {
  public:
    static const std::size_t size = n;
  };

  template <class T>
  class SizeHelper<std::vector<T>>
  {
  public:
    static const std::size_t size = 0;
  };


  template <typename VectorizedArrayType>
  class FreeEnergy
  {
  private:
    double A;
    double B;

  public:
    FreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c,
      const VectorType &         etas,
      const VectorizedArrayType &etaPower2Sum,
      const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
             B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                  (-4.0 * c + 8.0) * etaPower3Sum +
                  3.0 * (etaPower2Sum * etaPower2Sum));
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return f(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c,
          const VectorType &         etas,
          const VectorizedArrayType &etaPower2Sum,
          const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * (2.0 * c - 2.0) +
             2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
             B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return df_dc(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             const VectorizedArrayType &etaPower2Sum,
             unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return etai * B * 12.0 *
             (etai * (1.0 * c - 2.0) + (-c + 1.0) + etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return df_detai(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dc2(const VectorizedArrayType &c, const VectorType &etas) const
    {
      (void)etas;

      return 2.0 * A * (c * c) + 4.0 * A * c * (2.0 * c - 2.0) +
             2.0 * A * ((-c + 1.0) * (-c + 1.0)) + 2.0 * B;
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dcdetai(const VectorizedArrayType &c,
                const VectorType &         etas,
                unsigned int               index_i) const
    {
      (void)c;

      const auto &etai = etas[index_i];

      return B * 12.0 * etai * (etai - 1.0);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               const VectorizedArrayType &etaPower2Sum,
               unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                  24.0 * (etai * etai) + 12.0 * etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return d2f_detai2(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detaidetaj(const VectorizedArrayType &c,
                   const VectorType &         etas,
                   unsigned int               index_i,
                   unsigned int               index_j) const
    {
      (void)c;

      const auto &etai = etas[index_i];
      const auto &etaj = etas[index_j];

      return 24.0 * B * etai * etaj;
    }
  };
} // namespace Sintering

namespace Sintering
{
  namespace internal
  {
    template <typename Number>
    unsigned int
    n_blocks(const LinearAlgebra::distributed::Vector<Number> &)
    {
      return 1;
    }

    template <typename Number>
    unsigned int
    n_blocks(const LinearAlgebra::distributed::BlockVector<Number> &vector)
    {
      return vector.n_blocks();
    }

    template <typename Number>
    unsigned int
    n_blocks(
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &vector)
    {
      return vector.n_blocks();
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::Vector<Number> &vector,
          const unsigned int                          b)
    {
      AssertThrow(b == 0, ExcInternalError());
      return vector;
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::BlockVector<Number> &vector,
          const unsigned int                               b)
    {
      return vector.block(b);
    }

    template <typename Number>
    LinearAlgebra::distributed::Vector<Number> &
    block(LinearAlgebra::distributed::DynamicBlockVector<Number> &vector,
          const unsigned int                                      b)
    {
      return vector.block(b);
    }
  } // namespace internal

  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class OperatorBase : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    static const int dimension = dim;

    OperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const unsigned int                                  dof_index,
      const std::string                                   label        = "",
      const bool                                          matrix_based = false)
      : matrix_free(matrix_free)
      , constraints(constraints)
      , dof_index(dof_index)
      , label(label)
      , matrix_based(matrix_based)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(label != "")
      , do_timing(true)
    {}

    virtual ~OperatorBase() = default;

    virtual void
    clear()
    {
      this->system_matrix.clear();
      this->block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);

      constrained_indices.clear();
      constrained_values_src.clear();
    }

    virtual unsigned int
    n_components() const = 0;

    virtual unsigned int
    n_unique_components() const
    {
      return n_components();
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return matrix_free.get_dof_handler(dof_index);
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst, dof_index);
    }

    void
    initialize_dof_vector(BlockVectorType &dst) const
    {
      dst.reinit(this->n_components());
      for (unsigned int c = 0; c < this->n_components(); ++c)
        matrix_free.initialize_dof_vector(dst.block(c), dof_index);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::BlockVector<Number> &dst) const
    {
      dst.reinit(this->n_components());
      for (unsigned int c = 0; c < this->n_components(); ++c)
        matrix_free.initialize_dof_vector(dst.block(c), dof_index);
      dst.collect_sizes();
    }

    types::global_dof_index
    m() const
    {
      const auto &dof_handler = matrix_free.get_dof_handler(dof_index);

      if (dof_handler.get_fe().n_components() == 1)
        return dof_handler.n_dofs() * n_components();
      else
        return dof_handler.n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    bool
    set_timing(const bool do_timing) const
    {
      const bool old  = this->do_timing;
      this->do_timing = do_timing;
      return old;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult_", this->do_timing);

      if (matrix_based == false)
        {
          if (constrained_indices.empty())
            {
              const auto &constrained_dofs =
                this->matrix_free.get_constrained_dofs(this->dof_index);

              constrained_indices.resize(constrained_dofs.size());
              for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
                constrained_indices[i] = constrained_dofs[i];
              constrained_values_src.resize(this->n_components() *
                                            constrained_indices.size());
            }

          const bool is_scalar_dof_handler =
            this->matrix_free.get_dof_handler().get_fe().n_components() == 1;
          const unsigned int user_comp =
            is_scalar_dof_handler ? n_components() : 1;

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            for (unsigned int b = 0; b < user_comp; ++b)
              {
                constrained_values_src[i * user_comp + b] =
                  src.local_element(constrained_indices[i] * user_comp + b);
                const_cast<VectorType &>(src).local_element(
                  constrained_indices[i] * user_comp + b) = 0.;
              }

#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            for (unsigned int b = 0; b < user_comp; ++b)
              {
                const_cast<VectorType &>(src).local_element(
                  constrained_indices[i] * user_comp + b) =
                  constrained_values_src[i * user_comp + b];
                dst.local_element(constrained_indices[i] * user_comp + b) =
                  src.local_element(constrained_indices[i] * user_comp + b);
              }

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            {
              const_cast<VectorType &>(src).local_element(
                constrained_indices[i]) = constrained_values_src[i];
              dst.local_element(constrained_indices[i]) =
                constrained_values_src[i];
            }
        }
      else
        {
          system_matrix.vmult(dst, src);
        }
    }

    template <typename BlockVectorType_>
    void
    vmult(BlockVectorType_ &dst, const BlockVectorType_ &src) const
    {
      MyScope scope(this->timer, label + "::vmult", this->do_timing);

      if (matrix_based == false)
        {
          if (constrained_indices.empty())
            {
              const auto &constrained_dofs =
                this->matrix_free.get_constrained_dofs(this->dof_index);

              constrained_indices.resize(constrained_dofs.size());
              for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
                constrained_indices[i] = constrained_dofs[i];
              constrained_values_src.resize(this->n_components() *
                                            constrained_indices.size());
            }

          for (unsigned int b = 0; b < this->n_components(); ++b)
            for (unsigned int i = 0; i < constrained_indices.size(); ++i)
              {
                constrained_values_src[i + b * constrained_indices.size()] =
                  src.block(b).local_element(constrained_indices[i]);
                const_cast<BlockVectorType_ &>(src).block(b).local_element(
                  constrained_indices[i]) = 0.;
              }

#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

          for (unsigned int b = 0; b < this->n_components(); ++b)
            for (unsigned int i = 0; i < constrained_indices.size(); ++i)
              {
                const_cast<BlockVectorType_ &>(src).block(b).local_element(
                  constrained_indices[i]) =
                  constrained_values_src[i + b * constrained_indices.size()];
                dst.block(b).local_element(constrained_indices[i]) =
                  src.block(b).local_element(constrained_indices[i]);
              }
        }
      else
        {
          if (src_.size() == 0 || dst_.size() == 0)
            {
              const auto partitioner = get_system_partitioner();

              src_.reinit(partitioner);
              dst_.reinit(partitioner);
            }

          VectorTools::merge_components_fast(src, src_); // TODO
          this->vmult(dst_, src_);
          VectorTools::split_up_components_fast(dst_, dst); // TODO
        }
    }

    template <typename VectorType_>
    void
    Tvmult(VectorType_ &dst, const VectorType_ &src) const
    {
      AssertThrow(false, ExcNotImplemented());

      this->vmult(dst, src);
    }

    template <typename VectorType>
    void
    compute_inverse_diagonal(VectorType &diagonal) const
    {
      MyScope scope(this->timer, label + "::diagonal", this->do_timing);

      initialize_dof_vector(diagonal);

      Assert(internal::n_blocks(diagonal) == 1 ||
               matrix_free.get_dof_handler(dof_index).get_fe().n_components() ==
                 1,
             ExcInternalError());

#define OPERATION(c, d)                                                 \
  MatrixFreeTools::compute_diagonal(matrix_free,                        \
                                    diagonal,                           \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      for (unsigned int b = 0; b < internal::n_blocks(diagonal); ++b)
        for (auto &i : internal::block(diagonal, b))
          i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    std::shared_ptr<Utilities::MPI::Partitioner>
    get_system_partitioner() const
    {
      const auto partitioner_scalar =
        this->matrix_free.get_vector_partitioner(dof_index);

      IndexSet is(this->n_components());
      is.add_range(0, this->n_components());

      return std::make_shared<Utilities::MPI::Partitioner>(
        partitioner_scalar->locally_owned_range().tensor_product(is),
        partitioner_scalar->ghost_indices().tensor_product(is),
        partitioner_scalar->get_mpi_communicator());
    }

    TrilinosWrappers::SparseMatrix &
    get_system_matrix()
    {
      initialize_system_matrix();

      return system_matrix;
    }

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      initialize_system_matrix();

      return system_matrix;
    }

    void
    initialize_system_matrix() const
    {
      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer, label + "::matrix::sp", this->do_timing);

          system_matrix.clear();

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          DoFHandler<dim> dof_handler(
            this->matrix_free.get_dof_handler(dof_index).get_triangulation());
          dof_handler.distribute_dofs(
            FESystem<dim>(this->matrix_free.get_dof_handler(dof_index).get_fe(),
                          this->n_components()));

          constraints_for_matrix.clear();
          constraints_for_matrix.reinit(
            DoFTools::extract_locally_relevant_dofs(dof_handler));
          DoFTools::make_hanging_node_constraints(dof_handler,
                                                  constraints_for_matrix);
          constraints_for_matrix.close();

          dsp.reinit(dof_handler.locally_owned_dofs(),
                     dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          constraints_for_matrix,
                                          matrix_free.get_quadrature());
          dsp.compress();

          system_matrix.reinit(dsp);

          this->pcout << std::endl;
          this->pcout << "Create sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - NNZ: " << system_matrix.n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      label + "::matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          {
            system_matrix = 0.0; // clear existing content
          }

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    constraints_for_matrix,             \
                                    system_matrix,                      \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }
    }

    void
    clear_system_matrix() const
    {
      system_matrix.clear();
      block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    const TrilinosWrappers::SparsityPattern &
    get_sparsity_pattern() const
    {
      return dsp;
    }

    const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
    get_block_system_matrix() const
    {
      const bool system_matrix_is_empty = block_system_matrix.size() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer,
                        this->label + "::block_matrix::sp",
                        this->do_timing);

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          this->constraints,
                                          matrix_free.get_quadrature());
          dsp.compress();

          block_system_matrix.resize(this->n_components());
          for (unsigned int b = 0; b < this->n_components(); ++b)
            {
              block_system_matrix[b] =
                std::make_shared<TrilinosWrappers::SparseMatrix>();
              block_system_matrix[b]->reinit(dsp);
            }

          this->pcout << std::endl;
          this->pcout << "Create block sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - number of blocks: " << this->n_components()
                      << std::endl;
          this->pcout << " - NNZ:              "
                      << block_system_matrix[0]->n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      label + "::block_matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          for (unsigned int b = 0; b < this->n_components(); ++b)
            *block_system_matrix[b] = 0.0; // clear existing content

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    this->constraints,                  \
                                    block_system_matrix,                \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      return block_system_matrix;
    }

    void
    add_data_vectors(DataOut<dim> &               data_out,
                     const BlockVectorType &      vec,
                     const std::set<std::string> &fields_list) const
    {
#define OPERATION(c, d) \
  this->do_add_data_vectors<c, d>(data_out, vec, fields_list);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors(DataOut<dim> &               data_out,
                        const BlockVectorType &      vec,
                        const std::set<std::string> &fields_list) const
    {
      static_cast<const T &>(*this)
        .template do_add_data_vectors_kernel<n_comp, n_grains>(data_out,
                                                               vec,
                                                               fields_list);
    }

    virtual std::size_t
    memory_consumption() const
    {
      std::size_t result = 0;

      result += constraints_for_matrix.memory_consumption();
      result += system_matrix.memory_consumption();
      result += MyMemoryConsumption::memory_consumption(block_system_matrix);
      result += MyMemoryConsumption::memory_consumption(block_system_matrix);
      result += src_.memory_consumption();
      result += dst_.memory_consumption();
      result += MyMemoryConsumption::memory_consumption(constrained_indices);
      result += MyMemoryConsumption::memory_consumption(constrained_values_src);

      return result;
    }

  protected:
    template <int n_comp, int n_grains>
    void
    do_vmult_cell(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                   EvaluationFlags::EvaluationFlags::gradients);

      static_cast<const T &>(*this).template do_vmult_kernel<n_comp, n_grains>(
        phi);

      phi.integrate(EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      LinearAlgebra::distributed::BlockVector<Number> &      dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src,
      const std::pair<unsigned int, unsigned int> &          range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }


  protected:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const AffineConstraints<Number> &                   constraints;
    mutable AffineConstraints<Number>                   constraints_for_matrix;

    const unsigned int dof_index;

    const std::string label;
    const bool        matrix_based;

    mutable TrilinosWrappers::SparsityPattern dsp;
    mutable TrilinosWrappers::SparseMatrix    system_matrix;

    mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>>
      block_system_matrix;

    ConditionalOStream    pcout;
    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    mutable VectorType src_, dst_;

    mutable std::vector<unsigned int> constrained_indices;
    mutable std::vector<Number>       constrained_values_src;
  };



  template <int dim, typename VectorizedArrayType>
  struct SinteringOperatorData
  {
    using Number = typename VectorizedArrayType::value_type;

    // Choose MobilityScalar or MobilityTensorial here:
    static const bool use_tensorial_mobility =
#ifdef WITH_TENSORIAL_MOBILITY
      true;
#else
      false;
#endif

    using MobilityType =
      typename std::conditional<use_tensorial_mobility,
                                MobilityTensorial<dim, VectorizedArrayType>,
                                MobilityScalar<dim, VectorizedArrayType>>::type;

    SinteringOperatorData(const Number                      A,
                          const Number                      B,
                          const Number                      kappa_c,
                          const Number                      kappa_p,
                          std::shared_ptr<MobilityProvider> mobility_provider,
                          const unsigned int                integration_order)
      : free_energy(A, B)
      , kappa_c(kappa_c)
      , kappa_p(kappa_p)
      , time_data(integration_order)
      , mobility(mobility_provider)
    {}

    const FreeEnergy<VectorizedArrayType> free_energy;

    const Number kappa_c;
    const Number kappa_p;

    TimeIntegration::TimeIntegratorData<Number> time_data;

  public:
    Table<3, VectorizedArrayType> &
    get_nonlinear_values()
    {
      return nonlinear_values;
    }

    Table<3, VectorizedArrayType> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients()
    {
      return nonlinear_gradients;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
    }

    void
    set_n_components(const unsigned int number_of_components)
    {
      this->number_of_components = number_of_components;
    }

    unsigned int
    n_components() const
    {
      return number_of_components;
    }

    unsigned int
    n_grains() const
    {
      return number_of_components - 2;
    }

    void
    fill_quadrature_point_values(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &src,
      const bool save_op_gradients = false)
    {
      AssertDimension(src.n_blocks(), this->n_components());

      this->history_vector = src;
      this->history_vector.update_ghost_values();

      const unsigned n_cells             = matrix_free.n_cell_batches();
      const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

      nonlinear_values.reinit(
        {n_cells, n_quadrature_points, this->n_components()});

      nonlinear_gradients.reinit({n_cells,
                                  n_quadrature_points,
                                  use_tensorial_mobility || save_op_gradients ?
                                    this->n_components() :
                                    2});

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(matrix_free);

      src.update_ghost_values();

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          phi.reinit(cell);

          for (unsigned int c = 0; c < this->n_components(); ++c)
            {
              phi.read_dof_values_plain(src.block(c));
              phi.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  nonlinear_values(cell, q, c) = phi.get_value(q);

                  if (use_tensorial_mobility || (c < 2) || save_op_gradients)
                    nonlinear_gradients(cell, q, c) = phi.get_gradient(q);
                }
            }
        }

      src.zero_out_ghost_values();
    }

    virtual std::size_t
    memory_consumption() const
    {
      return nonlinear_values.memory_consumption() +
             nonlinear_gradients.memory_consumption();
    }

    const LinearAlgebra::distributed::DynamicBlockVector<Number> &
    get_history_vector() const
    {
      return history_vector;
    }

    void
    set_time(const double time)
    {
      mobility.update(time);
    }

    const MobilityType &
    get_mobility() const
    {
      return mobility;
    }

  private:
    MobilityType mobility;

    mutable Table<3, VectorizedArrayType> nonlinear_values;
    mutable Table<3, dealii::Tensor<1, dim, VectorizedArrayType>>
      nonlinear_gradients;

    unsigned int number_of_components;

    LinearAlgebra::distributed::DynamicBlockVector<Number> history_vector;
  };



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
        advection.reinit(cell, phi.get_matrix_free(), true, true);

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

                  value_result[0] += velocity * gradient[0];
                  value_result[ig + 2] += velocity * gradient[ig + 2];

                  const auto &velocity_der_c =
                    advection.get_velocity_derivative(ig,
                                                      phi.quadrature_point(q));

                  value_result[0] += velocity_der_c * c_grad * value[0];

                  value_result[ig + 2] +=
                    velocity_der_c * etas_grad[ig] * value[ig + 2];

                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      const auto &velocity_der_etaj =
                        advection.get_velocity_derivative(
                          ig, jg, phi.quadrature_point(q));

                      value_result[0] +=
                        velocity_der_etaj * gradient[jg + 2] * c_grad;

                      value_result[ig + 2] +=
                        velocity_der_etaj * gradient[jg + 2] * etas_grad[ig];
                    }
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
            advection.reinit(cell, matrix_free, true, false);

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
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
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
    do_update()
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

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim> &               data_out,
                               const BlockVectorType &      vec,
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
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
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

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          AdvectionOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = AdvectionOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    // Force, torque and grain volume
    static constexpr unsigned int n_comp_total = (dim == 3 ? 7 : 4);

    AdvectionOperator(
      const double                                           k,
      const double                                           cgb,
      const double                                           ceq,
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number> &             grain_tracker)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     AdvectionOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "advection_op",
          false)
      , k(k)
      , cgb(cgb)
      , ceq(ceq)
      , data(data)
      , grain_tracker(grain_tracker)
    {}

    ~AdvectionOperator()
    {}

    void
    evaluate_forces(const BlockVectorType &src,
                    AdvectionMechanism<dim, Number, VectorizedArrayType>
                      &advection_mechanism) const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

      std::pair<unsigned int, unsigned int> range{
        0, this->matrix_free.n_cell_batches()};

      src.update_ghost_values();

#define OPERATION(c, d) \
  do_evaluate_forces<c, d>(this->matrix_free, src, advection_mechanism, range);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      src.zero_out_ghost_values();
    }

    void
    evaluate_forces_derivatives(
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

      std::pair<unsigned int, unsigned int> range{
        0, this->matrix_free.n_cell_batches()};

#define OPERATION(c, d)                                     \
  do_evaluate_forces_derivatives<c, d>(this->matrix_free,   \
                                       advection_mechanism, \
                                       range);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    void
    do_update()
    {
      if (this->matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    unsigned int
    n_components() const override
    {
      return n_comp_total;
    }

    unsigned int
    n_grains() const
    {
      return data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      (void)n_grains;
      return n_comp_total;
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_forces_derivatives(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism,
      const std::pair<unsigned int, unsigned int> &         range) const
    {
      advection_mechanism.nullify_data_derivatives(grain_tracker.n_segments(),
                                                   n_grains);

      /*
            FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType>
         phi_sint( matrix_free, this->dof_index);
      */
      FECellIntegrator<dim,
                       advection_mechanism.n_comp_der_c,
                       Number,
                       VectorizedArrayType>
        phi_ft_dc(matrix_free, this->dof_index);

      using FECellIntegratorGrad =
        FECellIntegrator<dim,
                         advection_mechanism.n_comp_der_c,
                         Number,
                         VectorizedArrayType>;

      std::array<FECellIntegratorGrad, n_grains> phi_ft_dgrad_eta =
        create_array<n_grains>(
          FECellIntegratorGrad(matrix_free, this->dof_index));

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      /*
            FEValues<dim> fe_values(
              this->matrix_free.get_dof_handler(this->dof_index).get_fe(),
              this->matrix_free.get_quadrature(),
              update_values | update_gradients);
      */
      for (auto cell = range.first; cell < range.second; ++cell)
        {
          // phi_sint.reinit(cell);
          /*
          phi_sint.evaluate(EvaluationFlags::EvaluationFlags::values |
                            EvaluationFlags::EvaluationFlags::gradients);
          */
          /*
                    const auto icell0 = matrix_free.get_cell_iterator(cell, 0);
                    fe_values.reinit(icell0);

                    const auto sv00 = fe_values.shape_value(0, 0);
                    const auto sv01 = fe_values.shape_value(0, 1);
                    const auto sv02 = fe_values.shape_value(0, 2);
                    const auto sv03 = fe_values.shape_value(0, 3);

                    const auto sv10 = fe_values.shape_value(1, 0);
                    const auto sv11 = fe_values.shape_value(1, 1);
                    const auto sv12 = fe_values.shape_value(1, 2);
                    const auto sv13 = fe_values.shape_value(1, 3);

                    const auto sv20 = fe_values.shape_value(2, 0);
                    const auto sv21 = fe_values.shape_value(2, 1);
                    const auto sv22 = fe_values.shape_value(2, 2);
                    const auto sv23 = fe_values.shape_value(2, 3);

                    const auto sv30 = fe_values.shape_value(3, 0);
                    const auto sv31 = fe_values.shape_value(3, 1);
                    const auto sv32 = fe_values.shape_value(3, 2);
                    const auto sv33 = fe_values.shape_value(3, 3);
          */
          phi_ft_dc.reinit(cell);

          for (auto &phi_ft_i : phi_ft_dgrad_eta)
            phi_ft_i.reinit(cell);

          // const auto svalues = phi_sint.get_shape_info().data;
          // const auto sv00s = phi_sint.get_shape_info().get_shape_data(0, 0);
          // const auto sv10s = phi_sint.get_shape_info().get_shape_data(1, 0);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              Point<dim, VectorizedArrayType> rc;

              std::vector<std::pair<unsigned int, unsigned int>> segments(
                matrix_free.n_active_entries_per_cell_batch(cell));

              for (unsigned int i = 0; i < segments.size(); ++i)
                {
                  const auto icell = matrix_free.get_cell_iterator(cell, i);
                  const auto cell_index = icell->global_active_cell_index();

                  const unsigned int particle_id =
                    grain_tracker.get_particle_index(ig, cell_index);

                  if (particle_id != numbers::invalid_unsigned_int)
                    {
                      const auto grain_and_segment =
                        grain_tracker.get_grain_and_segment(ig, particle_id);

                      const auto &rc_i = grain_tracker.get_segment_center(
                        grain_and_segment.first, grain_and_segment.second);

                      for (unsigned int d = 0; d < dim; ++d)
                        rc[d][i] = rc_i[d];

                      segments[i] = grain_and_segment;
                    }
                  else
                    {
                      segments[i] =
                        std::make_pair(numbers::invalid_unsigned_int,
                                       numbers::invalid_unsigned_int);
                    }
                }

              for (unsigned int q = 0; q < phi_ft_dc.n_q_points; ++q)
                {
                  // Shape functions
                  // const auto phi_val  = phi_sint.get_value(q);
                  // const auto phi_grad = phi_sint.get_gradient(q);

                  // Nonlinear variables
                  const auto &value_lin    = nonlinear_values[cell][q];
                  const auto &gradient_lin = nonlinear_gradients[cell][q];

                  const auto &c          = value_lin[0];
                  const auto &eta_i      = value_lin[2 + ig];
                  const auto &eta_grad_i = gradient_lin[2 + ig];

                  const auto &r = phi_ft_dc.quadrature_point(q);

                  // Compute force and torque acting on grain i from each of the
                  // other grains
                  Tensor<1, n_grains, VectorizedArrayType> force_dgrad_eta;
                  Tensor<1, n_grains, Tensor<1, dim, VectorizedArrayType>>
                    torque_dgrad_eta;

                  Tensor<1, dim, VectorizedArrayType> force_dc;
                  moment_t<dim, VectorizedArrayType>  torque_dc;
                  torque_dc = 0.;

                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      if (ig != jg)
                        {
                          auto &eta_j      = value_lin[2 + jg];
                          auto &eta_grad_j = gradient_lin[2 + jg];

                          // Vector normal to the grain boundary
                          Tensor<1, dim, VectorizedArrayType> eta_grad_i_j =
                            eta_grad_i - eta_grad_j;

                          // Filter to detect grain boundary
                          auto etai_etaj = eta_i * eta_j;
                          etai_etaj      = compare_and_apply_mask<
                            SIMDComparison::greater_than>(etai_etaj,
                                                          cgb_lim,
                                                          etai_etaj,
                                                          zeros);

                          // Vector pointing from the grain center to the
                          // current qp point
                          const auto r_rc = (r - rc);

                          // Force derivative wrt grad_eta_j
                          auto df_dgrad_etaj = k * (c - ceq) * etai_etaj;

                          auto dt_dgrad_etaj = r_rc * df_dgrad_etaj;

                          force_dgrad_eta[ig] += df_dgrad_etaj;
                          torque_dgrad_eta[ig] += dt_dgrad_etaj;

                          force_dgrad_eta[jg] += -df_dgrad_etaj;
                          torque_dgrad_eta[jg] += -dt_dgrad_etaj;

                          // Force derivative wrt c
                          auto df_dc =
                            k * etai_etaj * eta_grad_i_j; // phi_val[0];
                          force_dc += df_dc;

                          // Torque derivative wrt c
                          torque_dc += cross_product(r_rc, df_dc);
                        }
                    }

                  Tensor<1,
                         advection_mechanism.n_comp_der_c,
                         VectorizedArrayType>
                    value_result_dc;

                  // Add force derivative wrt c
                  for (unsigned int d = 0; d < dim; ++d)
                    value_result_dc[d] = force_dc[d];

                  // Add torque derivative wrt c
                  for (unsigned int d = 0;
                       d < moment_s<dim, VectorizedArrayType>;
                       ++d)
                    value_result_dc[d + dim] = torque_dc[d];

                  phi_ft_dc.submit_value(value_result_dc, q);

                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      Tensor<1,
                             advection_mechanism.n_comp_der_grad_eta,
                             VectorizedArrayType>
                        value_result_dgrad_etaj;

                      value_result_dgrad_etaj[0] += force_dgrad_eta[jg];

                      for (unsigned int d = 0; d < dim; ++d)
                        value_result_dgrad_etaj[1 + d] +=
                          torque_dgrad_eta[jg][d];

                      phi_ft_dgrad_eta[jg].submit_value(value_result_dgrad_etaj,
                                                        q);
                    }
                }

              std::array<Tensor<1,
                                advection_mechanism.n_comp_der_grad_eta,
                                VectorizedArrayType>,
                         n_grains>
                force_torque_der_eta;

              for (unsigned int j = 0; j < n_grains; ++j)
                force_torque_der_eta[j] = phi_ft_dgrad_eta[j].integrate_value();

              Tensor<1, advection_mechanism.n_comp_der_c, VectorizedArrayType>
                force_torque_der_c = phi_ft_dc.integrate_value();

              for (unsigned int i = 0; i < segments.size(); ++i)
                {
                  const auto &grain_and_segment = segments[i];

                  if (grain_and_segment.first != numbers::invalid_unsigned_int)
                    {
                      for (unsigned int j = 0; j < n_grains; ++j)
                        for (unsigned int d = 0;
                             d < advection_mechanism.n_comp_der_grad_eta;
                             ++d)
                          advection_mechanism.grain_data_derivative(
                            grain_and_segment.first, grain_and_segment.second)
                            [j * advection_mechanism.n_comp_der_grad_eta + d] +=
                            force_torque_der_eta[j][d][i];

                      for (unsigned int d = 0;
                           d < advection_mechanism.n_comp_der_c;
                           ++d)
                        advection_mechanism.grain_data_derivative(
                          grain_and_segment.first, grain_and_segment.second)
                          [n_grains * advection_mechanism.n_comp_der_c + d] +=
                          force_torque_der_c[d][i];
                    }
                }
            }
        }

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    advection_mechanism.get_grains_data_derivatives().data(),
                    advection_mechanism.get_grains_data_derivatives().size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);
    }

    template <int n_comp, int n_grains>
    void
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const BlockVectorType &                               solution,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism,
      const std::pair<unsigned int, unsigned int> &         range) const
    {
      AssertDimension(advection_mechanism.n_comp_volume_force_torque,
                      n_comp_total);

      advection_mechanism.nullify_data(grain_tracker.n_segments(), n_grains);

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free, this->dof_index);

      FECellIntegrator<dim,
                       advection_mechanism.n_comp_volume_force_torque,
                       Number,
                       VectorizedArrayType>
        phi_ft(matrix_free, this->dof_index);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_ft.reinit(cell);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              Point<dim, VectorizedArrayType> rc;

              std::vector<std::pair<unsigned int, unsigned int>> segments(
                matrix_free.n_active_entries_per_cell_batch(cell));

              for (unsigned int i = 0; i < segments.size(); ++i)
                {
                  const auto icell = matrix_free.get_cell_iterator(cell, i);
                  const auto cell_index = icell->global_active_cell_index();

                  const unsigned int particle_id =
                    grain_tracker.get_particle_index(ig, cell_index);

                  if (particle_id != numbers::invalid_unsigned_int)
                    {
                      const auto grain_and_segment =
                        grain_tracker.get_grain_and_segment(ig, particle_id);

                      const auto &rc_i = grain_tracker.get_segment_center(
                        grain_and_segment.first, grain_and_segment.second);

                      for (unsigned int d = 0; d < dim; ++d)
                        rc[d][i] = rc_i[d];

                      segments[i] = grain_and_segment;
                    }
                  else
                    {
                      segments[i] =
                        std::make_pair(numbers::invalid_unsigned_int,
                                       numbers::invalid_unsigned_int);
                    }
                }

              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                {
                  const auto val  = phi_sint.get_value(q);
                  const auto grad = phi_sint.get_gradient(q);

                  auto &c          = val[0];
                  auto &eta_i      = val[2 + ig];
                  auto &eta_grad_i = grad[2 + ig];

                  const auto &r = phi_sint.quadrature_point(q);

                  Tensor<1,
                         advection_mechanism.n_comp_volume_force_torque,
                         VectorizedArrayType>
                                                      value_result;
                  Tensor<1, dim, VectorizedArrayType> force;
                  moment_t<dim, VectorizedArrayType>  torque;
                  torque = 0;

                  // Compute force and torque acting on grain i from each of the
                  // other grains
                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      if (ig != jg)
                        {
                          auto &eta_j      = val[2 + jg];
                          auto &eta_grad_j = grad[2 + jg];

                          // Vector normal to the grain boundary
                          Tensor<1, dim, VectorizedArrayType> dF =
                            eta_grad_i - eta_grad_j;

                          // Filter to detect grain boundary
                          auto etai_etaj = eta_i * eta_j;
                          etai_etaj      = compare_and_apply_mask<
                            SIMDComparison::greater_than>(etai_etaj,
                                                          cgb_lim,
                                                          etai_etaj,
                                                          zeros);

                          // Compute force component per cell
                          dF *= k * (c - ceq) * etai_etaj;

                          force += dF;

                          // Vector pointing from the grain center to the
                          // current qp point
                          const auto r_rc = (r - rc);

                          // Torque as cross product
                          // (scalar in 2D and vector in 3D)
                          torque += cross_product(r_rc, dF);
                        }
                    }

                  // Volume of grain i
                  value_result[0] = eta_i;

                  // Force acting on grain i
                  for (unsigned int d = 0; d < dim; ++d)
                    value_result[d + 1] = force[d];

                  // Torque acting on grain i
                  for (unsigned int d = 0;
                       d < moment_s<dim, VectorizedArrayType>;
                       ++d)
                    value_result[d + dim + 1] = torque[d];

                  phi_ft.submit_value(value_result, q);
                }

              const auto volume_force_torque = phi_ft.integrate_value();

              for (unsigned int i = 0; i < segments.size(); ++i)
                {
                  const auto &grain_and_segment = segments[i];

                  if (grain_and_segment.first != numbers::invalid_unsigned_int)
                    for (unsigned int d = 0;
                         d < advection_mechanism.n_comp_volume_force_torque;
                         ++d)
                      advection_mechanism.grain_data(
                        grain_and_segment.first, grain_and_segment.second)[d] +=
                        volume_force_torque[d][i];
                }
            }
        }

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    advection_mechanism.get_grains_data().data(),
                    advection_mechanism.get_grains_data().size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);
    }

    const double k;
    const double cgb;
    const double ceq;

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const GrainTracker::Tracker<dim, Number> &             grain_tracker;
  };

} // namespace Sintering
