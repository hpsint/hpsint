#pragma once

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

using namespace dealii;

template <typename Number>
class ProjectionOperatorBase
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  virtual ~ProjectionOperatorBase() = default;

  virtual void
  initialize_dof_vector(VectorType &vector) const = 0;

  virtual void
  initialize_dof_vector(BlockVectorType &vector) const = 0;

  virtual void
  compute_inverse_diagonal(VectorType &vector) const = 0;

  virtual void
  compute_inverse_diagonal(BlockVectorType &vector) const = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  virtual void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

  virtual void
  rhs(VectorType &dst) const = 0;

  virtual void
  rhs(BlockVectorType &dst) const = 0;

  virtual void
  vmult_local(BlockVectorType &dst, const BlockVectorType &src) const = 0;
};

template <int dim,
          int fe_degree,
          int n_q_points,
          int n_components,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
class ProjectionOperator : public ProjectionOperatorBase<Number>
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  ProjectionOperator(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    unsigned int                                        level = 2)
    : matrix_free(matrix_free)
    , level(level)
  {}

  void
  initialize_dof_vector(VectorType &vector) const override
  {
    matrix_free.initialize_dof_vector(vector);
  }

  void
  initialize_dof_vector(BlockVectorType &vector) const override
  {
    vector.reinit(n_components);

    for (unsigned int b = 0; b < n_components; ++b)
      matrix_free.initialize_dof_vector(vector.block(b));
  }

  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    matrix_free.initialize_dof_vector(diagonal);

    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &ProjectionOperator::do_vmult_cell,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  void
  compute_inverse_diagonal(BlockVectorType &vector) const override
  {
    vector.reinit(n_components);

    for (unsigned int b = 0; b < n_components; ++b)
      matrix_free.initialize_dof_vector(vector.block(b));

    vector = 1.0; // TODO
  }

  void
  do_vmult_cell(FEEvaluation<dim,
                             fe_degree,
                             n_q_points,
                             n_components,
                             Number,
                             VectorizedArrayType> &phi) const
  {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q_index = 0; q_index < phi.n_q_points; ++q_index)
      {
        phi.submit_value(phi.get_value(q_index), q_index);
        phi.submit_gradient(phi.get_gradient(q_index), q_index);
      }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    vmult_internal(dst, src);
  }

  void
  vmult(BlockVectorType &dst, const BlockVectorType &src) const override
  {
    vmult_internal(dst, src);
  }

  DEAL_II_ALWAYS_INLINE inline std::tuple<
    Tensor<1, n_components, VectorizedArrayType>,
    Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>>
  apply_q(const unsigned int                                  q,
          const Tensor<1, n_components, VectorizedArrayType> &value,
          const Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
            &gradient) const
  {
    (void)q;

    return {value, gradient};
  }

  template <typename VT>
  void
  vmult_internal(VT &dst, const VT &src) const
  {
    if (level == 4 || level == 5)
      matrix_free.template cell_loop<VT, VT>(
        [&](const auto &matrix_free,
            auto &      dst,
            const auto &src,
            const auto  range) {
          if (dim != 3)
            return;

          FEEvaluation<dim,
                       fe_degree,
                       n_q_points,
                       n_components,
                       Number,
                       VectorizedArrayType>
            phi(matrix_free);

          const bool use_fe_evaluation = false;

          constexpr unsigned int n_sub    = fe_degree > 0 ? fe_degree : 1;
          constexpr unsigned int nn       = n_sub + 1;
          constexpr unsigned int mm       = n_q_points > 0 ? n_q_points : 1;
          constexpr unsigned int n_points = Utilities::pow(mm, 3);
          constexpr unsigned int n_lanes  = VectorizedArrayType::size();
          VectorizedArrayType    dof_values_[n_components * nn * nn * nn] = {};
          VectorizedArrayType    values_quad[n_points + 1][n_components]  = {};
          VectorizedArrayType    gradients_z[n_points + 1][n_components]  = {};

          auto dof_values =
            use_fe_evaluation ? phi.begin_dof_values() : dof_values_;


          VectorizedArrayType ttmp_x0[n_components] = {};
          VectorizedArrayType ttmp_x1[n_components] = {};
          VectorizedArrayType ttmp_y0[n_components] = {};
          VectorizedArrayType ttmp_y1[n_components] = {};

          VectorizedArrayType tmp_x0[n_components] = {};
          VectorizedArrayType tmp_x1[n_components] = {};
          VectorizedArrayType tmp_y0[n_components] = {};
          VectorizedArrayType tmp_y1[n_components] = {};

          using value_type = Tensor<1, n_components, VectorizedArrayType>;
          using gradient_type =
            Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>;

          value_type    value_in_0    = {};
          gradient_type gradient_in_0 = {};

          const auto &              dof_info = matrix_free.get_dof_info();
          const VectorizedArrayType val0 =
            matrix_free.get_shape_info().data.front().shape_values[0];
          const VectorizedArrayType val1 =
            matrix_free.get_shape_info().data.front().shape_values[1];
          const VectorizedArrayType grad = Number(n_sub) / (val0 - val1);

          const auto src_vectors =
            internal::get_vector_data<n_components>(src, 0, false, 0, &dof_info)
              .first;
          const auto dst_vectors =
            internal::get_vector_data<n_components>(dst, 0, false, 0, &dof_info)
              .first;

          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              if (use_fe_evaluation == true)
                phi.reinit(cell);

              if (use_fe_evaluation == false)
                {
                  const unsigned int *dof_indices =
                    dof_info.dof_indices_interleaved.data() +
                    dof_info.row_starts[cell * n_lanes].first +
                    dof_info.component_dof_indices_offset[0][0] * n_lanes;
                  for (unsigned int iz = 0; iz < nn; ++iz)
                    {
                      for (unsigned int i = 0; i < nn * nn;
                           ++i, dof_indices += n_lanes)
                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          dof_values[comp * nn * nn * nn + iz * nn * nn + i]
                            .gather(src_vectors[comp]->begin(), dof_indices);
                    }
                }
              else
                {
                  phi.read_dof_values(src);
                }

              for (unsigned int iz = 0; iz < nn; ++iz)
                {
                  for (unsigned int comp = 0; comp < n_components; ++comp)
                    {
                      VectorizedArrayType temp[mm * nn] = {};
                      for (unsigned int i1 = 0; i1 < nn; ++i1)
                        {
                          temp[i1 * mm] =
                            val0 * dof_values[comp * nn * nn * nn +
                                              iz * nn * nn + i1 * nn] +
                            val1 * dof_values[comp * nn * nn * nn +
                                              iz * nn * nn + i1 * nn + 1];
                          for (unsigned int i0 = 0; i0 < n_sub - 1; ++i0)
                            {
                              const VectorizedArrayType t =
                                val0 *
                                dof_values[comp * nn * nn * nn + iz * nn * nn +
                                           i1 * nn + 1 + i0];
                              temp[i1 * mm + 2 * i0 + 1] =
                                t +
                                val1 * dof_values[comp * nn * nn * nn +
                                                  iz * nn * nn + i1 * nn + i0];
                              temp[i1 * mm + 2 * i0 + 2] =
                                t +
                                val1 *
                                  dof_values[comp * nn * nn * nn +
                                             iz * nn * nn + i1 * nn + i0 + 2];
                            }
                          temp[(i1 + 1) * mm - 1] =
                            val0 *
                              dof_values[comp * nn * nn * nn + iz * nn * nn +
                                         (i1 + 1) * nn - 1] +
                            val1 * dof_values[comp * nn * nn * nn +
                                              iz * nn * nn + (i1 + 1) * nn - 2];
                        }
                      for (unsigned int i1 = 0; i1 < mm; ++i1)
                        {
                          const unsigned int i = iz * mm * mm + i1;
                          values_quad[i][comp] =
                            val0 * temp[i1] + val1 * temp[i1 + mm];
                          for (unsigned int i0 = 0; i0 < n_sub - 1; ++i0)
                            {
                              const VectorizedArrayType t =
                                val0 * temp[i1 + (1 + i0) * mm];
                              values_quad[i + (2 * i0 + 1) * mm][comp] =
                                t + val1 * temp[i1 + i0 * mm];
                              values_quad[i + (2 * i0 + 2) * mm][comp] =
                                t + val1 * temp[i1 + (i0 + 2) * mm];
                            }
                          values_quad[i + (mm - 1) * mm][comp] =
                            val0 * temp[i1 + (nn - 1) * mm] +
                            val1 * temp[i1 + (nn - 2) * mm];
                        }
                    }
                }

              for (unsigned int i = 0; i < mm * mm; ++i)
                {
                  for (unsigned int comp = 0; comp < n_components; ++comp)
                    {
                      // work in reverse order because we alias between input
                      // and output array here
                      VectorizedArrayType tmp0 =
                        values_quad[i + (nn - 1) * mm * mm][comp];
                      VectorizedArrayType tmp1 =
                        values_quad[i + (nn - 2) * mm * mm][comp];
                      values_quad[i + (mm - 1) * mm * mm][comp] =
                        val0 * tmp0 + val1 * tmp1;
                      for (unsigned int i0 = n_sub - 1; i0 > 0; --i0)
                        {
                          const VectorizedArrayType t = val0 * tmp1;
                          values_quad[i + 2 * i0 * mm * mm][comp] =
                            t + val1 * tmp0;
                          const VectorizedArrayType gr =
                            grad *
                            (values_quad[i + (2 * i0 + 1) * mm * mm][comp] -
                             values_quad[i + 2 * i0 * mm * mm][comp]);
                          gradients_z[i + 2 * i0 * mm * mm][comp]       = gr;
                          gradients_z[i + (2 * i0 + 1) * mm * mm][comp] = gr;
                          tmp0                                          = tmp1;
                          tmp1 = values_quad[i + (i0 - 1) * mm * mm][comp];
                          values_quad[i + (2 * i0 - 1) * mm * mm][comp] =
                            t + val1 * tmp1;
                        }
                      values_quad[i][comp] = val0 * tmp1 + val1 * tmp0;
                      const VectorizedArrayType gr =
                        grad *
                        (values_quad[i + mm * mm][comp] - values_quad[i][comp]);
                      gradients_z[i + mm * mm][comp] = gr;
                      gradients_z[i][comp]           = gr;
                    }
                }

              const auto &mapping = matrix_free.get_mapping_info().cell_data[0];
              const auto  J_value = mapping.JxW_values[0];
              const auto  jacobian = mapping.jacobians[0][0];
              const Number quadrature_weight =
                mapping.descriptor[0].quadrature_weights[0];
              unsigned int offsets[n_sub * n_sub];
              for (unsigned int i1 = 0; i1 < n_sub; ++i1)
                for (unsigned int i0 = 0; i0 < n_sub; ++i0)
                  offsets[i1 * n_sub + i0] = i1 * mm * 2 + i0 * 2;

              if (level == 4)
                {
                  for (unsigned int iz = 0; iz < mm; ++iz)
                    for (unsigned int el = 0; el < n_sub * n_sub; ++el)
                      for (unsigned int comp = 0; comp < n_components; ++comp)
                        {
                          // Work for all 2x2 quadrature points of a single
                          // element in a plane to utilize register blocking
                          // between x and y derivatives
                          const unsigned int idx = iz * mm * mm + offsets[el];
                          const VectorizedArrayType gradient_y_0 =
                            grad * (values_quad[idx + mm][comp] -
                                    values_quad[idx][comp]);
                          const VectorizedArrayType gradient_y_1 =
                            grad * (values_quad[idx + mm + 1][comp] -
                                    values_quad[idx + 1][comp]);
                          const VectorizedArrayType gradient_x_0 =
                            grad * (values_quad[idx + 1][comp] -
                                    values_quad[idx][comp]);
                          const VectorizedArrayType gradient_x_1 =
                            grad * (values_quad[idx + mm + 1][comp] -
                                    values_quad[idx + mm][comp]);

                          const VectorizedArrayType JxW =
                            J_value * quadrature_weight;
                          // In the more general case with coupling between
                          // components, we would need to switch the loop over
                          // the inner 4 quadrature points (x-y plane) with the
                          // loop over components; this will cause register
                          // spills and increase the pressure on the L1 cache,
                          // but no other data transfer
                          values_quad[idx][comp] *= JxW;
                          values_quad[idx + 1][comp] *= JxW;
                          values_quad[idx + mm][comp] *= JxW;
                          values_quad[idx + mm + 1][comp] *= JxW;
                          gradients_z[idx][comp] *=
                            JxW * jacobian[2][2] * jacobian[2][2];
                          gradients_z[idx + 1][comp] *=
                            JxW * jacobian[2][2] * jacobian[2][2];
                          gradients_z[idx + mm][comp] *=
                            JxW * jacobian[2][2] * jacobian[2][2];
                          gradients_z[idx + mm + 1][comp] *=
                            JxW * jacobian[2][2] * jacobian[2][2];
                          // in the more general case, do operations on two y0,
                          // y1, x0, x1 slots each, so twice the work to be done
                          // here
                          const VectorizedArrayType y0 =
                            gradient_y_0 *
                            (JxW * jacobian[1][1] * jacobian[1][1]);
                          const VectorizedArrayType y1 =
                            gradient_y_1 *
                            (JxW * jacobian[1][1] * jacobian[1][1]);
                          const VectorizedArrayType x0 =
                            gradient_x_0 *
                            (JxW * jacobian[0][0] * jacobian[0][0]);
                          const VectorizedArrayType x1 =
                            gradient_x_1 *
                            (JxW * jacobian[0][0] * jacobian[0][0]);

                          // Now back to the interpolation operations, which are
                          // the general ones apart from 'x0 + x0' -> 'x0 + x2'
                          const VectorizedArrayType tmp_x0 = x0 + x0;
                          const VectorizedArrayType tmp_x1 = x1 + x1;
                          const VectorizedArrayType tmp_y0 = y0 + y0;
                          const VectorizedArrayType tmp_y1 = y1 + y1;
                          values_quad[idx][comp] -= grad * tmp_x0;
                          values_quad[idx][comp] -= grad * tmp_y0;
                          values_quad[idx + 1][comp] += grad * tmp_x0;
                          values_quad[idx + 1][comp] -= grad * tmp_y1;
                          values_quad[idx + mm][comp] -= grad * tmp_x1;
                          values_quad[idx + mm][comp] += grad * tmp_y0;
                          values_quad[idx + mm + 1][comp] += grad * tmp_x1;
                          values_quad[idx + mm + 1][comp] += grad * tmp_y1;
                        }
                }
              else
                {
                  for (unsigned int iz = 0; iz < mm; ++iz)
                    for (unsigned int el = 0; el < n_sub * n_sub; ++el)
                      {
                        const unsigned int idx = iz * mm * mm + offsets[el];
                        const VectorizedArrayType JxW =
                          J_value * quadrature_weight;

                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          {
                            ttmp_x0[comp] = (values_quad[idx + 1][comp] -
                                             values_quad[idx][comp]) *
                                            (grad * jacobian[0][0]);

                            ttmp_x1[comp] = (values_quad[idx + mm + 1][comp] -
                                             values_quad[idx + mm][comp]) *
                                            (grad * jacobian[0][0]);

                            ttmp_y0[comp] = (values_quad[idx + mm][comp] -
                                             values_quad[idx][comp]) *
                                            (grad * jacobian[1][1]);

                            ttmp_y1[comp] = (values_quad[idx + mm + 1][comp] -
                                             values_quad[idx + 1][comp]) *
                                            (grad * jacobian[1][1]);
                          }

                        for (unsigned int Q = 0; Q < 4; ++Q)
                          {
                            unsigned int idx_ = 0;

                            if (Q == 0)
                              idx_ = idx;
                            else if (Q == 1)
                              idx_ = idx + 1;
                            else if (Q == 2)
                              idx_ = idx + mm;
                            else
                              idx_ = idx + mm + 1;

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              {
                                gradient_in_0[comp][0] =
                                  (Q / 2 == 0) ? ttmp_x0[comp] : ttmp_x1[comp];
                                gradient_in_0[comp][1] =
                                  (Q % 2 == 0) ? ttmp_y0[comp] : ttmp_y1[comp];
                              }

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              value_in_0[comp] = values_quad[idx_][comp];

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              gradient_in_0[comp][2] =
                                gradients_z[idx_][comp] * jacobian[2][2];

                            const auto [value_out_0, gradient_out_0] =
                              apply_q(idx_, value_in_0, gradient_in_0);

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              values_quad[idx_][comp] = value_out_0[comp] * JxW;

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              gradients_z[idx_][comp] =
                                gradient_out_0[comp][2] *
                                (JxW * jacobian[2][2]);

                            for (unsigned int comp = 0; comp < n_components;
                                 ++comp)
                              {
                                const auto x0 = gradient_out_0[comp][0] *
                                                (JxW * jacobian[0][0]);

                                const auto y0 = gradient_out_0[comp][1] *
                                                (JxW * jacobian[1][1]);

                                // Now back to the interpolation operations
                                if (Q == 0)
                                  {
                                    tmp_x0[comp] = x0;
                                    tmp_y0[comp] = y0;
                                  }
                                else if (Q == 1)
                                  {
                                    tmp_x0[comp] += x0;
                                    tmp_y1[comp] = y0;
                                  }
                                else if (Q == 2)
                                  {
                                    tmp_x1[comp] = x0;
                                    tmp_y0[comp] += y0;
                                  }
                                else
                                  {
                                    tmp_x1[comp] += x0;
                                    tmp_y1[comp] += y0;
                                  }
                              }
                          }

                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          values_quad[idx][comp] +=
                            (-tmp_x0[comp] - tmp_y0[comp]) * grad;

                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          values_quad[idx + 1][comp] +=
                            (+tmp_x0[comp] - tmp_y1[comp]) * grad;

                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          values_quad[idx + mm][comp] +=
                            (-tmp_x1[comp] + tmp_y0[comp]) * grad;

                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          values_quad[idx + mm + 1][comp] +=
                            (+tmp_x1[comp] + tmp_y1[comp]) * grad;
                      }
                }

              for (unsigned int i = 0; i < mm * mm; ++i)
                {
                  for (unsigned int comp = 0; comp < n_components; ++comp)
                    {
                      const VectorizedArrayType tmp =
                        gradients_z[i][comp] + gradients_z[i + mm * mm][comp];
                      VectorizedArrayType tmp0 =
                        values_quad[i][comp] - grad * tmp;
                      VectorizedArrayType tmp1 =
                        values_quad[i + mm * mm][comp] + grad * tmp;
                      values_quad[i][comp] = val0 * tmp0 + val1 * tmp1;
                      for (unsigned int i0 = 1; i0 < n_sub; ++i0)
                        {
                          VectorizedArrayType sum = val1 * tmp0 + val0 * tmp1;
                          const VectorizedArrayType tmp =
                            gradients_z[i + 2 * i0 * mm * mm][comp] +
                            gradients_z[i + (2 * i0 + 1) * mm * mm][comp];
                          tmp0 = values_quad[i + 2 * i0 * mm * mm][comp] -
                                 grad * tmp;
                          tmp1 = values_quad[i + (2 * i0 + 1) * mm * mm][comp] +
                                 grad * tmp;
                          sum += val0 * tmp0;
                          sum += val1 * tmp1;
                          values_quad[i + i0 * mm * mm][comp] = sum;
                        }
                      values_quad[i + n_sub * mm * mm][comp] =
                        val1 * tmp0 + val0 * tmp1;
                    }
                }

              for (unsigned int iz = 0; iz < nn; ++iz)
                {
                  for (unsigned int comp = 0; comp < n_components; ++comp)
                    {
                      VectorizedArrayType temp[mm * nn] = {};
                      for (unsigned int i1 = 0; i1 < mm; ++i1)
                        {
                          const unsigned int  i    = iz * mm * mm + i1;
                          VectorizedArrayType tmp0 = values_quad[i][comp];
                          VectorizedArrayType tmp1 = values_quad[i + mm][comp];
                          temp[i1]                 = val0 * tmp0 + val1 * tmp1;
                          for (unsigned int i0 = 1; i0 < n_sub; ++i0)
                            {
                              VectorizedArrayType sum =
                                val1 * tmp0 + val0 * tmp1;
                              tmp0 = values_quad[i + 2 * i0 * mm][comp];
                              tmp1 = values_quad[i + (2 * i0 + 1) * mm][comp];
                              sum += val0 * tmp0;
                              sum += val1 * tmp1;
                              temp[i1 + i0 * mm] = sum;
                            }
                          temp[i1 + n_sub * mm] = val1 * tmp0 + val0 * tmp1;
                        }
                      for (unsigned int i1 = 0; i1 < nn; ++i1)
                        {
                          VectorizedArrayType tmp0 = temp[i1 * mm];
                          VectorizedArrayType tmp1 = temp[i1 * mm + 1];
                          dof_values[comp * nn * nn * nn + iz * nn * nn +
                                     i1 * nn]      = val0 * tmp0 + val1 * tmp1;
                          for (unsigned int i0 = 1; i0 < n_sub; ++i0)
                            {
                              VectorizedArrayType sum =
                                val1 * tmp0 + val0 * tmp1;
                              tmp0 = temp[i1 * mm + 2 * i0];
                              tmp1 = temp[i1 * mm + 2 * i0 + 1];
                              sum += val0 * tmp0;
                              sum += val1 * tmp1;
                              dof_values[comp * nn * nn * nn + iz * nn * nn +
                                         i1 * nn + i0] = sum;
                            }
                          dof_values[comp * nn * nn * nn + iz * nn * nn +
                                     i1 * nn + n_sub] =
                            val1 * tmp0 + val0 * tmp1;
                        }
                    }
                }

              if (use_fe_evaluation == false)
                {
                  const unsigned int *dof_indices =
                    dof_info.dof_indices_interleaved.data() +
                    dof_info.row_starts[cell * n_lanes].first +
                    dof_info.component_dof_indices_offset[0][0] * n_lanes;

                  for (unsigned int iz = 0; iz < nn; ++iz)
                    {
                      for (unsigned int i = 0; i < nn * nn;
                           ++i, dof_indices += n_lanes)
                        for (unsigned int comp = 0; comp < n_components; ++comp)
                          {
#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS < 512
                            for (unsigned int v = 0; v < n_lanes; ++v)
                              (*dst_vectors[comp])
                                .local_element(dof_indices[v]) +=
                                dof_values[comp * nn * nn * nn + iz * nn * nn +
                                           i][v];
#else
                            // only use gather in case there is also scatter.
                            VectorizedArrayType tmp = {};
                            tmp.gather(dst_vectors[comp]->begin(), dof_indices);
                            tmp += dof_values[comp * nn * nn * nn +
                                              iz * nn * nn + i];
                            tmp.scatter(dof_indices,
                                        dst_vectors[comp]->begin());
#endif
                          }
                    }
                }
              else
                {
                  phi.distribute_local_to_global(dst);
                }
            }
        },
        dst,
        src,
        true);
    else
      matrix_free.template cell_loop<VT, VT>(
        [this](const auto &matrix_free,
               auto &      dst,
               const auto &src,
               const auto  range) {
          FEEvaluation<dim,
                       fe_degree,
                       n_q_points,
                       n_components,
                       Number,
                       VectorizedArrayType>
            phi(matrix_free, range);

          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              phi.reinit(cell);

              op_local(phi, dst, src);
            }
        },
        dst,
        src,
        true);
  }

  void
  vmult_local(BlockVectorType &dst, const BlockVectorType &src) const override
  {
    matrix_free.template cell_loop<BlockVectorType, BlockVectorType>(
      [this](
        const auto &matrix_free, auto &dst, const auto &src, const auto range) {
        FEEvaluation<dim, fe_degree, n_q_points, 1, Number, VectorizedArrayType>
          phi(matrix_free, range);

        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int b = 0; b < n_components; ++b)
              op_local(phi, dst.block(b), src.block(b));
          }
      },
      dst,
      src,
      true);
  }

private:
  template <int nc, typename VT>
  void
  op_local(
    FEEvaluation<dim, fe_degree, n_q_points, nc, Number, VectorizedArrayType>
      &       phi,
    VT &      dst,
    const VT &src) const
  {
    if (level == 0)
      phi.read_dof_values(src);
    else
      phi.gather_evaluate(src,
                          EvaluationFlags::values | EvaluationFlags::gradients);

    if (level == 2)
      for (unsigned int q_index = 0; q_index < phi.n_q_points; ++q_index)
        {
          phi.submit_value(phi.get_value(q_index), q_index);
          phi.submit_gradient(phi.get_gradient(q_index), q_index);
        }
    else if (level == 3)
      {
        const auto nqp =
          n_q_points != 0 ? Utilities::pow(n_q_points, dim) : phi.n_q_points;

        Tensor<1, nc, Tensor<1, dim, VectorizedArrayType>> grad;

        auto values_quad    = phi.begin_values();
        auto gradients_quad = phi.begin_gradients();

        const auto &mapping = matrix_free.get_mapping_info().cell_data[0];
        const auto  quadrature_weights =
          mapping.descriptor[0].quadrature_weights.begin();
        const auto J_value  = mapping.JxW_values.begin();
        const auto jacobian = mapping.jacobians[0].begin();

        for (unsigned int q_index = 0; q_index < nqp; ++q_index)
          {
            const VectorizedArrayType JxW =
              J_value[0] * quadrature_weights[q_index];

            // values
            {
              // get_value
              Tensor<1, nc, VectorizedArrayType> val;
              for (unsigned int comp = 0; comp < nc; ++comp)
                val[comp] = values_quad[comp * nqp + q_index];

              for (unsigned int comp = 0; comp < nc; ++comp)
                values_quad[comp * nqp + q_index] = val[comp] * JxW;
            }

            // gradients
            {
              std::array<VectorizedArrayType, dim> jac;

              for (unsigned int d = 0; d < dim; ++d)
                jac[d] = jacobian[0][d][d];

              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int comp = 0; comp < nc; ++comp)
                  grad[comp][d] =
                    gradients_quad[(comp * dim + d) * nqp + q_index] * jac[d];

              for (unsigned int d = 0; d < dim; ++d)
                {
                  const VectorizedArrayType factor = jac[d] * JxW;
                  for (unsigned int comp = 0; comp < nc; ++comp)
                    gradients_quad[(comp * dim + d) * nqp + q_index] =
                      grad[comp][d] * factor;
                }
            }
          }
      }

    if (level == 0)
      phi.distribute_local_to_global(dst);
    else
      phi.integrate_scatter(EvaluationFlags::values |
                              EvaluationFlags::gradients,
                            dst);
  }

public:
  void
  rhs(VectorType &dst) const override
  {
    rhs_internal(dst);
  }

  void
  rhs(BlockVectorType &dst) const override
  {
    rhs_internal(dst);
  }

  template <typename VT>
  void
  rhs_internal(VT &dst) const
  {
    double dummy = 0.;
    matrix_free.template cell_loop<VT, double>(
      [](const auto &matrix_free, auto &dst, const auto &, const auto range) {
        FEEvaluation<dim,
                     fe_degree,
                     n_q_points,
                     n_components,
                     Number,
                     VectorizedArrayType>
          phi(matrix_free, range);

        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);

            for (unsigned int q_index = 0; q_index < phi.n_q_points; ++q_index)
              {
                if constexpr (n_components == 1)
                  phi.submit_value(VectorizedArrayType(1.0), q_index);
                else
                  AssertThrow(false, ExcNotImplemented());
              }

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      dummy,
      true);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  level;
};
