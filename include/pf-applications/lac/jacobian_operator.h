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

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <functional>
#include <string>

namespace NonLinearSolvers
{
  using namespace dealii;

  template <typename Number>
  class JacobianBase : public Subscriptor
  {
  public:
    using value_type  = Number;
    using vector_type = LinearAlgebra::distributed::Vector<Number>;
    using VectorType  = vector_type;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    virtual void
    reinit(const VectorType &vec) = 0;

    virtual void
    reinit(const BlockVectorType &vec) = 0;
  };

  template <typename Number, typename OperatorType>
  class JacobianWrapper : public JacobianBase<Number>
  {
  public:
    using value_type      = typename JacobianBase<Number>::value_type;
    using vector_type     = typename JacobianBase<Number>::vector_type;
    using VectorType      = typename JacobianBase<Number>::VectorType;
    using BlockVectorType = typename JacobianBase<Number>::BlockVectorType;

    JacobianWrapper(const OperatorType &op)
      : op(op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      op.vmult(dst, src);
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      op.vmult(dst, src);
    }

    void
    reinit(const VectorType &) override
    {
      // TODO: nothing to do, since done elsewhere dirictly on the
      // operator
    }

    void
    reinit(const BlockVectorType &) override
    {
      // TODO: nothing to do, since done elsewhere dirictly on the
      // operator
    }

  private:
    const OperatorType &op;
  };



  template <typename T, typename VectorType>
  using evaluate_nonlinear_residual_with_pre_post_t =
    decltype(std::declval<const T>().evaluate_nonlinear_residual(
      std::declval<VectorType &>(),
      std::declval<const VectorType &>(),
      std::declval<
        const std::function<void(const unsigned int, const unsigned int)>>(),
      std::declval<
        const std::function<void(const unsigned int, const unsigned int)>>()));

  template <typename T, typename VectorType>
  constexpr bool has_evaluate_nonlinear_residual_with_pre_post =
    dealii::internal::is_supported_operation<
      evaluate_nonlinear_residual_with_pre_post_t,
      T,
      VectorType>;



  template <typename Number, typename OperatorType>
  class JacobianFree : public JacobianBase<Number>
  {
  public:
    using value_type      = typename JacobianBase<Number>::value_type;
    using vector_type     = typename JacobianBase<Number>::vector_type;
    using VectorType      = typename JacobianBase<Number>::VectorType;
    using BlockVectorType = typename JacobianBase<Number>::BlockVectorType;

    JacobianFree(const OperatorType &op,
                 const std::string   step_length_algo = "pw")
      : op(op)
      , step_length_algo(step_length_algo)
    {}

    void
    vmult(VectorType &, const VectorType &) const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "jacobi_free::vmult");

      // 1) determine step length
      value_type h = 1e-8;

      if (step_length_algo == "bs") // cost: 2r + global reduction
        {
          const auto       ua        = u * src;
          const auto       a_l1_norm = src.l1_norm();
          const auto       a_l2_norm = src.l2_norm();
          const value_type u_min     = 1e-6;

          if (a_l2_norm == 0)
            h = 0.0;
          else if (std::abs(ua) > u_min * a_l1_norm)
            h *= ua / (a_l2_norm * a_l2_norm);
          else
            h *= u_min * (ua >= 0.0 ? 1.0 : -1.0) * a_l1_norm /
                 (a_l2_norm * a_l2_norm);
        }
      else if (step_length_algo == "pw") // cost: 1r + global reduction
        {
          const auto a_l2_norm = src.l2_norm();

          if (a_l2_norm == 0)
            h = 0.0;
          else
            h *= std::sqrt(1.0 + u_l2_norm) / a_l2_norm;
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      if (h == 0.0)
        {
          dst = 0.0;
        }
      else
        {
          if constexpr (has_evaluate_nonlinear_residual_with_pre_post<
                          OperatorType,
                          BlockVectorType>)
            {
              const auto  delta     = h;
              const auto  delta_inv = 1.0 / delta;
              auto       &tmp       = u;
              const auto &res       = residual_u;

              const auto pre_op = [&](const auto begin, const auto end) {
                for (unsigned int b = 0; b < Sintering::internal::n_blocks(dst);
                     ++b)
                  {
                    const auto dst_ptr =
                      Sintering::internal::block(dst, b).begin();
                    const auto src_ptr =
                      Sintering::internal::block(src, b).begin();
                    const auto tmp_ptr =
                      Sintering::internal::block(tmp, b).begin();

                    DEAL_II_OPENMP_SIMD_PRAGMA
                    for (std::size_t i = begin; i < end; ++i)
                      {
                        // zero out destination vector
                        dst_ptr[i] = 0.0;

                        // perturb
                        tmp_ptr[i] += delta * src_ptr[i];
                      }
                  }
              };

              const auto post_op = [&](const auto begin, const auto end) {
                for (unsigned int b = 0; b < Sintering::internal::n_blocks(dst);
                     ++b)
                  {
                    const auto dst_ptr =
                      Sintering::internal::block(dst, b).begin();
                    const auto src_ptr =
                      Sintering::internal::block(src, b).begin();
                    const auto tmp_ptr =
                      Sintering::internal::block(tmp, b).begin();
                    const auto res_ptr =
                      Sintering::internal::block(res, b).begin();

                    DEAL_II_OPENMP_SIMD_PRAGMA
                    for (std::size_t i = begin; i < end; ++i)
                      {
                        // compute finite difference
                        dst_ptr[i] = (dst_ptr[i] - res_ptr[i]) * delta_inv;

                        // clean up
                        tmp_ptr[i] -= delta * src_ptr[i];
                      }
                  }
              };

              op.template evaluate_nonlinear_residual<1>(dst,
                                                         tmp,
                                                         pre_op,
                                                         post_op);
            }
          else
            {
              // 2) approximate Jacobian-vector product
              //    cost: 4r + 2w

              // 2a) perturb linerization point -> pre
              u.add(h, src);

              // 2b) evalute residual
              op.template evaluate_nonlinear_residual<1>(dst, u);

              // 2c) take finite difference -> post
              dst.add(-1.0, residual_u);
              dst *= 1.0 / h;

              // 2d) cleanup -> post
              u.add(-h, src);
            }
        }
    }

    void
    reinit(const VectorType &) override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    reinit(const BlockVectorType &u) override
    {
      MyScope scope(timer, "jacobi_free::reinit");

      this->u = u;

      if (step_length_algo == "pw")
        this->u_l2_norm = u.l2_norm();

      this->residual_u.reinit(u);
      op.template evaluate_nonlinear_residual<1>(this->residual_u, u);
    }

  private:
    const OperatorType &op;
    const std::string   step_length_algo;

    mutable BlockVectorType u;
    mutable BlockVectorType residual_u;

    mutable value_type u_l2_norm;

    mutable MyTimerOutput timer;
  };
} // namespace NonLinearSolvers
