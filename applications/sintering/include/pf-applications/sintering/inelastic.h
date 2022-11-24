#pragma once

#include <deal.II/base/tensor.h>

#include <pf-applications/numerics/functions.h>

#include <pf-applications/sintering/sintering_data.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/structural/tools.h>

namespace Sintering
{
  using namespace dealii;
  template <int dim, typename Number, typename VectorizedArrayType>
  class InelasticStrains
  {
  public:
    InelasticStrains(
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data,
      const double                                           rho,
      const double                                           time_start = 0.0)
      : sintering_data(sintering_data)
      , rho(rho)
      , time_start(time_start)
    {}

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    flux_eps_dot(const VectorizedArrayType &c,
                 const VectorTypeValue &    etas,
                 const unsigned int         etas_size,
                 const VectorTypeGradient & etas_grad,
                 const VectorizedArrayType &div_gb,
                 const VectorizedArrayType &div_vol) const
    {
      if (sintering_data.get_time() < time_start)
        return Tensor<2, dim, VectorizedArrayType>();

      const auto v_val = v(c);

      const auto eps_inelastic =
        v_val * chi(etas, etas_size, etas_grad, div_gb, div_vol);

      return eps_inelastic;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    flux_eps_dot_dc(const VectorizedArrayType &c,
                    const VectorTypeValue &    etas,
                    const unsigned int         etas_size,
                    const VectorTypeGradient & etas_grad,
                    const VectorizedArrayType &div_gb,
                    const VectorizedArrayType &div_vol) const
    {
      if (sintering_data.get_time() < time_start)
        return Tensor<2, dim, VectorizedArrayType>();

      const auto dvdc = dv_dc(c);

      const auto eps_inelastic =
        chi(etas, etas_size, etas_grad, div_gb, div_vol);

      auto NNS = eps_inelastic;
      NNS *= dvdc;

      return NNS;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    flux_eps_dot_detai(const VectorizedArrayType &c,
                       const VectorTypeValue &    etas,
                       const unsigned int         etas_size,
                       const VectorTypeGradient & etas_grad,
                       const VectorizedArrayType &div_gb,
                       const VectorizedArrayType &div_vol,
                       const unsigned int         index_i) const
    {
      if (sintering_data.get_time() < time_start)
        return Tensor<2, dim, VectorizedArrayType>();

      const auto dgdetai = dg_detai(etas, etas_size, index_i);

      const auto n = gb_norm(etas, etas_size, etas_grads);
      const auto k = gb_norm_detai(etas, etas_size, etas_grads, index_i);

      auto KN_NK = outer_product(k, n) + outer_product(n, k);

      KN_NK *= rho * div_gb * g_val * fac_gb;

      auto NN = outer_product(n, n);
      NN *= rho * div_gb * dgdetai * fac_gb;

      auto D = diagonal_matrix<dim>();
      D *= dgdetai * div_vol / volume_denominator * fac_vol;

      const auto v_val = v(c);

      auto NNS = KN_NK + NN + D;
      NNS *= v_val;

      return NNS;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    flux_eps_dot_ddiv_gb(const VectorizedArrayType &c,
                         const VectorTypeValue &    etas,
                         const unsigned int         etas_size) const
    {
      if (sintering_data.get_time() < time_start)
        return Tensor<2, dim, VectorizedArrayType>();

      const auto v_val = v(c);
      const auto g_val = g(etas, etas_size);

      auto NNS = diagonal_matrix<dim>();
      NNS *= v_val * g_val * rho;

      return Structural::apply_l(NNS);
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    flux_eps_dot_ddiv_vol(const VectorizedArrayType &c,
                          const VectorTypeValue &    etas,
                          const unsigned int         etas_size) const
    {
      if (sintering_data.get_time() < time_start)
        return Tensor<2, dim, VectorizedArrayType>();

      const auto v_val = v(c);
      const auto g_val = g(etas, etas_size);

      auto NNS = diagonal_matrix<dim>();
      NNS *= v_val * (1. - g_val) / volume_denominator;

      return NNS;
    }

  private:
    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<2, dim, VectorizedArrayType>
    chi(const VectorTypeValue &    etas,
        const unsigned int         etas_size,
        const VectorTypeGradient & etas_grad,
        const VectorizedArrayType &div_gb,
        const VectorizedArrayType &div_vol) const
    {
      const auto normal = gb_norm(etas, etas_size, etas_grads);

      const auto g_val = g(etas, etas_size);

      auto eps_dot_gb = outer_product(normal, normal);

      eps_dot_gb *= rho * div_gb * g_val * fac_gb;

      const auto eps_dot_vol = diagonal_matrix<dim>(
        -div_vol / volume_denominator * (1 - g_val) * fac_vol);

      const auto eps_dot_inelastic = eps_dot_gb + eps_dot_vol;

      return eps_dot_inelastic;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<1, dim, VectorizedArrayType>
    gb_norm(const VectorTypeValue &   etas,
            const unsigned int        etas_size,
            const VectorTypeGradient &etas_grad) const
    {
      Tensor<1, dim, VectorTypeValue> n;

      for (unsigned int i = 0; i < etas_size; i++)
        for (unsigned int j = i + 1; j < etas_size; j++)
          {
            Tensor<1, dim, VectorizedArrayType> eta_grad_diff =
              (etas_grad[i]) - (etas_grad[j]);
            Tensor<1, dim, VectorizedArrayType> neta =
              unit_vector(eta_grad_diff);

            n += neta;
          }

      n *= gb_indicator(etas, etas_size);

      return n;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    Tensor<1, dim, VectorizedArrayType>
    gb_norm_detai(const VectorTypeValue &   etas,
                  const unsigned int        etas_size,
                  const VectorTypeGradient &etas_grad,
                  const unsigned            index_i) const
    {
      (void)etas;
      (void)etas_size;
      (void)etas_grad;
      (void)index_i;

      // The term id disabled temporary

      Tensor<1, dim, VectorTypeValue> k;

      return k;
    }

    template <typename VectorTypeValue>
    VectorizedArrayType
    gb_indicator(const VectorTypeValue &etas,
                 const unsigned int     etas_size) const
    {
      for (unsigned int i = 0; i < etas_size; i++)
        for (unsigned int j = 0; j < etas_size; j++)
          if (i != j)
            gb += etas[i] * etas[j];

      gb = compare_and_apply_mask<SIMDComparison::less_than>(
        gb, VectorizedArrayType(0.1), VectorizedArrayType(0.0), gb);

      return gb;
    }

    template <typename VectorTypeValue>
    VectorizedArrayType
    g(const VectorTypeValue &etas, const unsigned int etas_size) const
    {
      VectorizedArrayType g_val(0.);

      for (unsigned int i = 0; i < etas_size; i++)
        for (unsigned int j = i + 1; j < etas_size; j++)
          g_val += std::pow(2 * 4 * etas[i] * etas[j], 4);
      // other version
      // g_val += 4 * etas[i] * etas[j];

      phi = compare_and_apply_mask<SIMDComparison::less_than>(
        phi, VectorizedArrayType(0.0), VectorizedArrayType(0.0), phi);
      phi = compare_and_apply_mask<SIMDComparison::greater_than>(
        phi, VectorizedArrayType(1.0), VectorizedArrayType(1.0), phi);

      return g_val;
    }

    template <typename VectorTypeValue>
    VectorizedArrayType
    dg_detai(const VectorTypeValue &etas,
             const unsigned int     etas_size,
             unsigned int           index_i) const
    {
      VectorizedArrayType dg_val(0.);

      for (unsigned int j = 0; j < etas_size; j++)
        if (j != index_i)
          dg_val += etas[j];

      dg_val *= 4.;

      return dg_val;
    }

    VectorizedArrayType
    v(const VectorizedArrayType &c) const
    {
      auto v_val = c * c;

      v_val = compare_and_apply_mask<SIMDComparison::greater_than>(
        v_val, VectorizedArrayType(1.0), VectorizedArrayType(1.0), v_val);

      return v_val;
    }

    VectorizedArrayType
    dv_dc(const VectorizedArrayType &c) const
    {
      return 2 * c;
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data;

    const double rho;
    const double time_start;

    const double fac_gb             = 1.; // 1 - enabled, 0 - disabled
    const double fac_vol            = 1.; // 1 - enabled, 0 - disabled
    const double volume_denominator = 3.; // 3. - initial value
  };

} // namespace Sintering
