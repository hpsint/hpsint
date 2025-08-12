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

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <array>
#include <utility>

namespace KKS
{
  using namespace dealii;

  enum Medium
  {
    vidx,
    sidx
  };

  inline unsigned int
  classidx(const unsigned int a)
  {
    return a != 0;
  }

  template <typename Number>
  using Matrix2x2 = std::array<std::array<Number, 2>, 2>;

  template <typename Number>
  class PropMatrix
  {
  private:
    Matrix2x2<Number> values;
    Number            prefac;

  public:
    PropMatrix(Matrix2x2<Number> values_in, Number prefac_in)
      : values(values_in)
      , prefac(prefac_in)
    {}

    Number
    operator()(unsigned int i, unsigned int j) const
    {
      return prefac * values[classidx(i)][classidx(j)];
    }
  };

  int
  linf(const unsigned int a)
  {
    return a == 0 ? 1 : -1;
  }

  template <long unsigned int n_phi, typename VectorizedArrayType>
  VectorizedArrayType
  hfunc2_0(const std::array<VectorizedArrayType, n_phi> &phi)
  {
    const auto phisqsum = std::accumulate(phi.begin(),
                                          phi.end(),
                                          VectorizedArrayType(0),
                                          [](const auto &sum, const auto &val) {
                                            return sum + val * val;
                                          });

    return phi[0] * phi[0] / phisqsum;
  }

  template <typename Number,
            std::array<Number, 2> k,
            std::array<Number, 2> c_0,
            long unsigned int     n_phi,
            typename VectorizedArrayType>
  VectorizedArrayType
  calc_ca2(const VectorizedArrayType                    &c,
           const std::array<VectorizedArrayType, n_phi> &phio,
           const unsigned int                            a)
  {
    const auto   idx    = static_cast<unsigned>(a == 0);
    const Number prefac = linf(a);

    const auto hv = hfunc2_0(phio);

    const std::array<VectorizedArrayType, 2> h{{hv, 1. - hv}};

    const auto chi =
      c_0[KKS::vidx] * k[KKS::vidx] - c_0[KKS::sidx] * k[KKS::sidx];

    const auto nom = c * k[idx] + prefac * chi * h[idx];

    const auto denom = k[0] * h[1] + k[1] * h[0];

    return nom / denom;
  }

  template <typename Number,
            std::array<Number, 2> k,
            std::array<Number, 2> c_0,
            int                   dim,
            long unsigned int     n_phi,
            typename VectorizedArrayType>
  Tensor<1, dim, VectorizedArrayType>
  calc_grad_ca2(
    const VectorizedArrayType                                    &c,
    const Tensor<1, dim, VectorizedArrayType>                    &c_grad,
    const std::array<VectorizedArrayType, n_phi>                 &phio,
    const std::array<Tensor<1, dim, VectorizedArrayType>, n_phi> &phio_grad,
    const unsigned int                                            a)
  {
    const auto   idx    = static_cast<unsigned>(a == 0);
    const Number prefac = linf(a);

    const auto hv = hfunc2_0(phio);

    const std::array<VectorizedArrayType, 2> h{{hv, 1. - hv}};

    VectorizedArrayType                 phisqsum(0.0);
    Tensor<1, dim, VectorizedArrayType> phisqsum_grad;
    for (unsigned int i = 0; i < n_phi; ++i)
      {
        phisqsum += phio[i] * phio[i];
        phisqsum_grad += phio[i] * phio_grad[i];
      }

    const Tensor<1, dim, VectorizedArrayType> hv_grad =
      2. * phio[0] / (phisqsum * phisqsum) *
      (phio_grad[0] * phisqsum - phio[0] * phisqsum_grad);

    const auto chi =
      c_0[KKS::vidx] * k[KKS::vidx] - c_0[KKS::sidx] * k[KKS::sidx];

    const auto nom        = c * k[idx] + prefac * chi * h[idx];
    const auto nom_grad   = c_grad * k[idx] - chi * hv_grad;
    const auto denom      = k[0] * h[1] + k[1] * h[0];
    const auto denom_grad = (k[1] - k[0]) * hv_grad;

    return (nom_grad * denom - nom * denom_grad) / (denom * denom);
  }

  template <long unsigned int n_phi,
            typename Number,
            typename VectorizedArrayType>
  VectorizedArrayType
  Zw(const PropMatrix<Number>                     &Z,
     const std::array<VectorizedArrayType, n_phi> &phi)
  {
    VectorizedArrayType res(0.);

    for (unsigned int i = 0; i < n_phi; ++i)
      for (unsigned int j = i + 1; j < n_phi; ++j)
        res += Z(i, j) * std::pow(phi[i], 2.) * std::pow(phi[j], 2.);

    return res;
  }

  template <long unsigned int n_phi,
            typename Number,
            typename VectorizedArrayType>
  VectorizedArrayType
  Zwp(const PropMatrix<Number>                     &Z,
      const std::array<VectorizedArrayType, n_phi> &phi,
      const unsigned int                            j)
  {
    VectorizedArrayType res(0.);

    for (unsigned int i = 0; i < n_phi; ++i)
      if (i != j)
        res += Z(i, j) * std::pow(phi[i], 2.);

    res *= 2. * phi[j];

    return res;
  }

  template <long unsigned int n_phi, typename VectorizedArrayType>
  VectorizedArrayType
  w(const std::array<VectorizedArrayType, n_phi> &phi)
  {
    VectorizedArrayType res(0.);

    for (unsigned int i = 0; i < n_phi; ++i)
      for (unsigned int j = i + 1; j < n_phi; ++j)
        res += std::pow(phi[i], 2.) * std::pow(phi[j], 2.);

    return res;
  }

  template <long unsigned int n_phi, typename VectorizedArrayType>
  VectorizedArrayType
  wp(const std::array<VectorizedArrayType, n_phi> &phi, const unsigned int j)
  {
    VectorizedArrayType res(0.);

    for (unsigned int i = 0; i < n_phi; ++i)
      if (i != j)
        res += std::pow(phi[i], 2.);

    res *= 2. * phi[j];

    return res;
  }

  template <typename Number, typename VectorizedArrayType>
  VectorizedArrayType
  calc_ga(const VectorizedArrayType &ca, const Number ki, const Number c_0i)
  {
    return 0.5 * ki * std::pow(ca - c_0i, 2.);
  }

  template <typename Number, typename VectorizedArrayType>
  VectorizedArrayType
  calc_mu(const VectorizedArrayType &ca, const Number ki, const Number c_0i)
  {
    return ki * (ca - c_0i);
  }

  template <typename Number,
            std::array<Number, 2> k,
            std::array<Number, 2> c_0,
            bool                  do_couple_phi_c,
            int                   dim,
            long unsigned int     n_phi,
            typename VectorizedArrayType>
  std::pair<VectorizedArrayType, Tensor<1, dim, VectorizedArrayType>>
  dFdphi(const PropMatrix<Number>                                     &A,
         const PropMatrix<Number>                                     &B,
         const std::array<VectorizedArrayType, n_phi>                 &phi,
         const std::array<Tensor<1, dim, VectorizedArrayType>, n_phi> &phi_grad,
         const VectorizedArrayType                                    &c,
         const unsigned int                                            j,
         const Number div_eps   = 1e-18,
         const Number threshold = 1e-18)
  {
    const auto w_val   = w(phi);
    const auto wp_val  = wp(phi, j);
    const auto Bw_val  = Zw(B, phi);
    const auto Bwp_val = Zwp(B, phi, j);
    const auto Aw_val  = Zw(A, phi);
    const auto Awp_val = Zwp(A, phi, j);

    const VectorizedArrayType zeroes(0.0), ones(1.0), w_threshold(threshold);
    const auto w_fix_val = compare_and_apply_mask<SIMDComparison::less_than>(
      std::abs(w_val), w_threshold, ones, w_val);
    auto w_div_val = compare_and_apply_mask<SIMDComparison::less_than>(
      std::abs(w_val), w_threshold, ones, std::abs(w_val) + div_eps);
    w_div_val = 1. / (w_div_val * w_div_val);

    const auto q_val = 0.5 * w_fix_val + 1. / 12. +
                       std::accumulate(phi.begin(),
                                       phi.end(),
                                       VectorizedArrayType(0),
                                       [](const auto &sum, const auto &val) {
                                         return sum + std::pow(val, 4.) / 4. -
                                                std::pow(val, 3.) / 3.;
                                       });

    const auto qp_val =
      0.5 * wp_val + std::pow(phi[j], 3.) - std::pow(phi[j], 2.);

    const auto phi_grad_sq_sum =
      std::accumulate(phi_grad.begin(),
                      phi_grad.end(),
                      VectorizedArrayType(0),
                      [](const auto &sum, const auto &val) {
                        return sum + val.norm_square();
                      });

    VectorizedArrayType res_val(0.0);

    // Term 1 - verified
    const auto Ap_val = (Awp_val * w_fix_val - Aw_val * wp_val) * w_div_val;
    res_val += 0.5 * Ap_val * phi_grad_sq_sum;

    // Terms 2 and 3 - verified
    res_val += ((Bwp_val * w_fix_val - Bw_val * wp_val) * q_val +
                (Bw_val / w_fix_val) * qp_val * std::pow(w_fix_val, 2.)) *
               w_div_val;

    // Term 4 - seems verified
    if constexpr (do_couple_phi_c)
      {
        const auto phi_sq_sum =
          std::accumulate(phi.begin(),
                          phi.end(),
                          VectorizedArrayType(0),
                          [](const auto &sum, const auto &val) {
                            return sum + val * val;
                          });

        // Evaluate psi's
        const auto ca_0 = calc_ca2<Number, k, c_0>(c, phi, 0);
        const auto ca_1 = calc_ca2<Number, k, c_0>(c, phi, 1);
        const auto mu   = calc_mu(ca_0, k[0], c_0[0]);

        std::array<VectorizedArrayType, 2> psi{
          {(calc_ga(ca_0, k[0], c_0[0]) - mu * ca_0),
           (calc_ga(ca_1, k[1], c_0[1]) - mu * ca_1)}};

        for (unsigned int i = 0; i < n_phi; ++i)
          res_val +=
            (-2. * phi[j] * phi[i] * phi[i] / (phi_sq_sum * phi_sq_sum)) *
            psi[classidx(i)];

        res_val += (2. * phi[j] / phi_sq_sum) * psi[classidx(j)];
      }

    // Divergence term - verified
    const auto A_val = Aw_val / w_fix_val;

    Tensor<1, dim, VectorizedArrayType> res_grad = A_val * phi_grad[j];

    return std::make_pair(std::move(res_val), std::move(res_grad));
  }

  template <int dim, typename VectorizedArrayType>
  Tensor<1, dim, VectorizedArrayType>
  central_flux(const VectorizedArrayType                 &u_m,
               const VectorizedArrayType                 &u_p,
               const Tensor<1, dim, VectorizedArrayType> &normal)
  {
    const auto flux = (u_p - u_m) * normal;
    return flux;
  }

} // namespace KKS