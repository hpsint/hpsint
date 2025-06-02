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

#include <deal.II/matrix_free/tensor_product_kernels.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_sintering_base.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim,
            int fe_degree,
            int n_q_points_1D,
            int n_comp,
            typename Number,
            typename VectorizedArrayType>
  class SinteringOperatorGenericQuad
  {
  public:
    using FECellIntegratorType = FEEvaluation<dim,
                                              fe_degree,
                                              n_q_points_1D,
                                              n_comp,
                                              Number,
                                              VectorizedArrayType>;

    DEAL_II_ALWAYS_INLINE inline SinteringOperatorGenericQuad(
      const FECellIntegratorType                                 &phi,
      const FreeEnergy<VectorizedArrayType>                      &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>      &data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      Tensor<1, n_comp, VectorizedArrayType> *gradient_buffer)
      : phi(phi)
      , cell(phi.get_current_cell_index())
      , lin_value(data.get_nonlinear_values(cell))
      , lin_gradient(data.get_nonlinear_gradients(cell))
      , free_energy(free_energy)
      , mobility(data.get_mobility())
      , kappa_c(data.kappa_c)
      , kappa_p(data.kappa_p)
      , weight(data.time_data.get_primary_weight())
      , L(mobility.Lgb())
      , advection_data(cell, advection, data)
      , gradient_buffer(gradient_buffer)
    {
      if (gradient_buffer != nullptr)
        {
          const auto &shape_values =
            phi.get_shape_info().data[0].shape_gradients_collocation;

          using EvaluatorTypeData = Tensor<1, n_comp, VectorizedArrayType>;
          using EvaluatorTypeCoefficients = typename std::remove_reference_t<
            decltype(shape_values)>::value_type;

          dealii::internal::EvaluatorTensorProduct<
            dealii::internal::EvaluatorVariant::evaluate_general,
            dim,
            n_q_points_1D,
            n_q_points_1D,
            EvaluatorTypeData,
            EvaluatorTypeCoefficients>
            phi;

          // gradient x-direction
          phi.template apply<0, true, false>(
            shape_values.data(),
            reinterpret_cast<const EvaluatorTypeData *>(lin_value),
            gradient_buffer + 0 * FECellIntegratorType::static_n_q_points);

          if constexpr (dim >= 2) // gradient y-direction
            phi.template apply<1, true, false>(
              shape_values.data(),
              reinterpret_cast<const EvaluatorTypeData *>(lin_value),
              gradient_buffer + 1 * FECellIntegratorType::static_n_q_points);

          if constexpr (dim >= 3) // gradient z-direction
            phi.template apply<2, true, false>(
              shape_values.data(),
              reinterpret_cast<const EvaluatorTypeData *>(lin_value),
              gradient_buffer + 2 * FECellIntegratorType::static_n_q_points);
        }
    }

    DEAL_II_ALWAYS_INLINE inline std::tuple<
      typename FECellIntegratorType::value_type,
      typename FECellIntegratorType::gradient_type>
    operator()(
      const unsigned int                                  q,
      const typename FECellIntegratorType::value_type    &value,
      const typename FECellIntegratorType::gradient_type &gradient) const
    {
      typename FECellIntegratorType::value_type    value_result;
      typename FECellIntegratorType::gradient_type gradient_result;

      constexpr int n_grains = n_comp - 2;

      const VectorizedArrayType *lin_etas_value = &lin_value[0] + 2;

      const auto free_energy_eval =
        free_energy.template eval<EnergySecond, n_grains>(lin_value);


      // 1) process c row
      value_result[0] = value[0] * weight;

      if (gradient_buffer == nullptr)
        {
          gradient_result[0] = mobility.apply_M_derivative(
            &lin_value[0], &lin_gradient[0], n_grains, &value[0], &gradient[0]);
        }
      else
        {
          const auto lin_gradient = this->get_lin_gradient(q);

          gradient_result[0] = mobility.apply_M_derivative(
            &lin_value[0], &lin_gradient[0], n_grains, &value[0], &gradient[0]);
        }



      // 2) process mu row
      value_result[1] = -value[1] + free_energy_eval.d2f_dc2() * value[0];

      for (unsigned int ig = 0; ig < n_grains; ++ig)
        value_result[1] +=
          free_energy_eval.d2f_dcdetai(lin_etas_value[ig]) * value[ig + 2];

      gradient_result[1] = kappa_c * gradient[0];



      // 3) process eta rows
      for (unsigned int ig = 0; ig < n_grains; ++ig)
        {
          value_result[ig + 2] +=
            value[ig + 2] * weight +
            L *
              (free_energy_eval.d2f_dcdetai(lin_etas_value[ig]) * value[0] +
               free_energy_eval.d2f_detai2(lin_etas_value[ig]) * value[ig + 2]);

          gradient_result[ig + 2] = L * kappa_p * gradient[ig + 2];
        }

      free_energy.apply_d2f_detaidetaj(
        L, lin_etas_value, n_grains, &value[0] + 2, &value_result[0] + 2);



      // 4) add advection contributations -> influences c AND etas
      if (!advection_data.empty())
        for (unsigned int ig = 0; ig < n_grains; ++ig)
          if (advection_data.has_velocity(ig))
            {
              const auto &velocity_ig =
                advection_data.get_velocity(ig, phi.quadrature_point(q));

              gradient_result[0] -= velocity_ig * value[0];
              gradient_result[2 + ig] -= velocity_ig * value[2 + ig];
            }



      lin_value += 2 + n_grains;
      lin_gradient +=
        2 +
        (SinteringOperatorData<dim,
                               VectorizedArrayType>::use_tensorial_mobility ?
           n_grains :
           0);

      return {value_result, gradient_result};
    }

  private:
    const FECellIntegratorType                                &phi;
    const unsigned int                                         cell;
    const VectorizedArrayType mutable                         *lin_value;
    const dealii::Tensor<1, dim, VectorizedArrayType> mutable *lin_gradient;
    const FreeEnergy<VectorizedArrayType>                     &free_energy;
    const typename SinteringOperatorData<dim, VectorizedArrayType>::MobilityType
                &mobility;
    const Number kappa_c;
    const Number kappa_p;
    const Number weight;
    const Number L;
    const AdvectionVelocityData<dim, Number, VectorizedArrayType>
      advection_data;

    Tensor<1, n_comp, VectorizedArrayType> *gradient_buffer;

    DEAL_II_ALWAYS_INLINE inline Tensor<1,
                                        n_comp,
                                        Tensor<1, dim, VectorizedArrayType>>
    get_lin_gradient(const unsigned int q) const
    {
      const auto &mapping_data =
        phi.get_matrix_free().get_mapping_info().cell_data[0];

      const unsigned int offsets = mapping_data.data_index_offsets[cell];

      const auto &jacobian = mapping_data.jacobians[0][offsets];

      Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>> gradient;

      for (unsigned int c = 0; c < n_comp; ++c)
        for (unsigned int d = 0; d < dim; ++d)
          gradient[c][d] =
            jacobian[d][d] *
            gradient_buffer[q + d * FECellIntegratorType::static_n_q_points][c];

      return gradient;
    }
  };

  template <int dim, typename VectorizedArrayType, int with_time_derivative>
  class SinteringOperatorGenericResidualQuad
  {
  public:
    using Number = typename VectorizedArrayType::value_type;

    DEAL_II_ALWAYS_INLINE inline SinteringOperatorGenericResidualQuad(
      const FreeEnergy<VectorizedArrayType>                 &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const AlignedVector<VectorizedArrayType>              &buffer)
      : free_energy(free_energy)
      , mobility(data.get_mobility())
      , kappa_c(data.kappa_c)
      , kappa_p(data.kappa_p)
      , weight(data.time_data.get_primary_weight())
      , L(mobility.Lgb())
      , buffer(buffer)
    {}

    template <typename T1, typename T2>
    DEAL_II_ALWAYS_INLINE inline std::tuple<T1, T2>
    operator()(const unsigned int q, const T1 &value, const T2 &gradient) const
    {
      T1 value_result;
      T2 gradient_result;

      if constexpr (!std::is_same<T1, VectorizedArrayType>::value)
        if constexpr (T1::dimension >= 2)
          {
            const unsigned int n_comp   = T1::dimension;
            const unsigned int n_grains = n_comp - 2;

            const VectorizedArrayType *etas_value = &value[0] + 2;
            const Tensor<1, dim, VectorizedArrayType> *etas_gradient =
              &gradient[0] + 2;

            const auto free_energy_eval =
              free_energy.template eval<EnergyFirst, n_grains>(value);

            // 1) process c row
            if (with_time_derivative >= 1)
              value_result[0] = value[0] * weight;
            if (with_time_derivative == 2)
              value_result[0] += buffer[n_comp * q];

            gradient_result[0] = mobility.apply_M(value[0],
                                                  etas_value,
                                                  n_grains,
                                                  gradient[0],
                                                  etas_gradient,
                                                  gradient[1]);

            // 2) process mu row
            value_result[1]    = -value[1] + free_energy_eval.df_dc();
            gradient_result[1] = kappa_c * gradient[0];


            // 3) process eta rows
            for (unsigned int ig = 0; ig < n_grains; ++ig)
              {
                value_result[2 + ig] =
                  L * free_energy_eval.df_detai(etas_value[ig]);

                if (with_time_derivative >= 1)
                  value_result[ig + 2] += value[ig + 2] * weight;
                if (with_time_derivative == 2)
                  value_result[2 + ig] += buffer[n_comp * q + 2 + ig];

                gradient_result[2 + ig] = L * kappa_p * gradient[2 + ig];
              }
          }

      return {value_result, gradient_result};
    }

  private:
    const FreeEnergy<VectorizedArrayType> &free_energy;
    const typename SinteringOperatorData<dim, VectorizedArrayType>::MobilityType
                &mobility;
    const Number kappa_c;
    const Number kappa_p;
    const Number weight;
    const Number L;

    const AlignedVector<VectorizedArrayType> &buffer;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperatorGeneric
    : public SinteringOperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = SinteringOperatorGeneric<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorGeneric(
      const MatrixFree<dim, Number, VectorizedArrayType>         &matrix_free,
      const AffineConstraints<Number>                            &constraints,
      const FreeEnergy<VectorizedArrayType>                      &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>      &data,
      const TimeIntegration::SolutionHistory<BlockVectorType>    &history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const bool                                                  matrix_based,
      const bool use_tensorial_mobility_gradient_on_the_fly)
      : SinteringOperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          free_energy,
          data,
          history,
          matrix_based)
      , advection(advection)
      , use_tensorial_mobility_gradient_on_the_fly(
          use_tensorial_mobility_gradient_on_the_fly)
    {}

    ~SinteringOperatorGeneric()
    {}

    template <typename... Arg>
    static T
    create(
      const MatrixFree<dim, Number, VectorizedArrayType>      &matrix_free,
      const AffineConstraints<Number>                         &constraints,
      const FreeEnergy<VectorizedArrayType>                   &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>   &sintering_data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &solution_history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType>
                &advection_mechanism,
      const bool matrix_based,
      const bool use_tensorial_mobility_gradient_on_the_fly = false,
      Arg &&...)
    {
      return T(matrix_free,
               constraints,
               free_energy,
               sintering_data,
               solution_history,
               advection_mechanism,
               matrix_based,
               use_tensorial_mobility_gradient_on_the_fly);
    }

    template <unsigned int with_time_derivative = 2>
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

      if (this->data.get_component_table().size(0) == 0)
        {
#define OPERATION(c, d)                                           \
  MyMatrixFreeTools::cell_loop_wrapper(                           \
    this->matrix_free,                                            \
    &SinteringOperatorGeneric::                                   \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    pre_operation_to_be_used,                                     \
    post_operation);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
          MyMatrixFreeTools::cell_loop_wrapper(
            this->matrix_free,
            &SinteringOperatorGeneric::do_evaluate_nonlinear_residual_nt<
              with_time_derivative>,
            this,
            dst,
            src,
            pre_operation_to_be_used,
            post_operation);
        }
    }

    void
    vmult_internal(VectorType &dst, const VectorType &src) const override
    {
      OperatorBase<dim, Number, VectorizedArrayType, T>::vmult(dst, src);
    }

    void
    do_vmult_range_no_template(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      std::vector<
        std::shared_ptr<FEEvaluationData<dim, VectorizedArrayType, false>>>
        phis(this->n_grains() + 1);

      AlignedVector<VectorizedArrayType> gradient_buffer;

      if (SinteringOperatorData<dim,
                                VectorizedArrayType>::use_tensorial_mobility &&
          use_tensorial_mobility_gradient_on_the_fly)
        {
          gradient_buffer.resize_fast(
            dim * this->n_components() *
            FECellIntegrator<dim, 1, Number, VectorizedArrayType>::
              static_n_q_points);
        }

      for (unsigned int i = 0; i <= this->n_grains(); ++i)
        {
          const unsigned int n_comp_nt = i + 2;
#define OPERATION(n_comp, dummy)                                 \
  (void)dummy;                                                   \
  phis[i] = std::make_shared<                                    \
    FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>>( \
    matrix_free, this->dof_index);
          EXPAND_OPERATIONS_N_COMP_NT(OPERATION);
#undef OPERATION
        }

      std::vector<const VectorType *> src_view;
      std::vector<VectorType *>       dst_view;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          // CH blocks
          for (unsigned int b = 0; b < 2; ++b)
            {
              src_view.push_back(&src.block(b));
              dst_view.push_back(&dst.block(b));
            }

          // relevant AC blocks
          for (const auto b : this->data.get_relevant_grains(cell))
            {
              src_view.push_back(&src.block(b + 2));
              dst_view.push_back(&dst.block(b + 2));
            }

          const unsigned int n_comp_nt = src_view.size();

#define OPERATION(n_comp, dummy)                                               \
  (void)dummy;                                                                 \
  constexpr unsigned int n_grains = n_comp - 2;                                \
                                                                               \
  auto &phi =                                                                  \
    static_cast<FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &>( \
      *phis[n_grains]);                                                        \
                                                                               \
  phi.reinit(cell);                                                            \
  phi.read_dof_values(src_view);                                               \
  phi.evaluate(EvaluationFlags::EvaluationFlags::values |                      \
               EvaluationFlags::EvaluationFlags::gradients);                   \
                                                                               \
  static_cast<const T &>(*this).template do_vmult_kernel<n_comp, n_grains>(    \
    phi, gradient_buffer.empty() ? nullptr : gradient_buffer.data());          \
                                                                               \
  phi.integrate(EvaluationFlags::EvaluationFlags::values |                     \
                EvaluationFlags::EvaluationFlags::gradients);                  \
  phi.distribute_local_to_global(dst_view);
          EXPAND_OPERATIONS_N_COMP_NT(OPERATION);
#undef OPERATION

          src_view.clear();
          dst_view.clear();
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      AlignedVector<VectorizedArrayType> gradient_buffer;

      if (SinteringOperatorData<dim,
                                VectorizedArrayType>::use_tensorial_mobility &&
          use_tensorial_mobility_gradient_on_the_fly)
        {
          gradient_buffer.resize_fast(
            dim * this->n_components() *
            FECellIntegrator<dim, 1, Number, VectorizedArrayType>::
              static_n_q_points);
        }

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(
              phi, gradient_buffer.empty() ? nullptr : gradient_buffer.data());

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    void
    vmult_internal(BlockVectorType       &dst,
                   const BlockVectorType &src) const override
    {
      if (this->data.get_component_table().size(0) == 0)
        {
#define OPERATION(c, d)                              \
  MyMatrixFreeTools::cell_loop_wrapper(              \
    this->matrix_free,                               \
    &SinteringOperatorGeneric::do_vmult_range<c, d>, \
    this,                                            \
    dst,                                             \
    src,                                             \
    true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
          MyMatrixFreeTools::cell_loop_wrapper(
            this->matrix_free,
            &SinteringOperatorGeneric::do_vmult_range_no_template,
            this,
            dst,
            src,
            true);
        }
    }

    void
    vmult_internal(
      LinearAlgebra::distributed::BlockVector<Number>       &dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src) const override
    {
      OperatorBase<dim, Number, VectorizedArrayType, T>::vmult(dst, src);
    }

    unsigned int
    n_components() const override
    {
      return this->data.n_components();
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

    template <int n_comp, int n_grains, int fe_degree, int n_q_points>
    void
    do_vmult_kernel(FEEvaluation<dim,
                                 fe_degree,
                                 n_q_points,
                                 n_comp,
                                 Number,
                                 VectorizedArrayType> &phi,
                    VectorizedArrayType *gradient_buffer = nullptr) const
    {
      AssertDimension(n_comp - 2, n_grains);

      const SinteringOperatorGenericQuad<dim,
                                         fe_degree,
                                         n_q_points,
                                         n_comp,
                                         Number,
                                         VectorizedArrayType>
        quad_op(phi,
                this->free_energy,
                this->data,
                this->advection,
                reinterpret_cast<Tensor<1, n_comp, VectorizedArrayType> *>(
                  gradient_buffer));

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          const auto [value_result, gradient_result] =
            quad_op(q, value, gradient);

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim>                &data_out,
                               const BlockVectorType       &vec,
                               const std::set<std::string> &fields_list) const
    {
      // Output from the parent
      SinteringOperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>::
        template do_add_data_vectors_kernel<n_comp, n_grains>(data_out,
                                                              vec,
                                                              fields_list);

      // Possible output options - at the moment only velocity
      enum OutputFields
      {
        FieldVelocity
      };

      constexpr unsigned int n_data_variants = 1;

      const std::array<std::tuple<std::string, OutputFields, unsigned int>,
                       n_data_variants>
        possible_entries = {{{"vel", FieldVelocity, n_grains * dim}}};

      // Get active entries to output
      const auto [entries_mask, n_entries] =
        this->get_vector_output_entries_mask(possible_entries, fields_list);

      if (n_entries == 0)
        return;

      std::vector<VectorType> data_vectors(n_entries);

      for (auto &data_vector : data_vectors)
        this->matrix_free.initialize_dof_vector(data_vector, this->dof_index);

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> fe_eval(
        this->matrix_free, this->dof_index);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, 1, Number, VectorizedArrayType>
          inverse_mass_matrix(fe_eval);

      AlignedVector<VectorizedArrayType> buffer(fe_eval.n_q_points * n_entries);

      std::vector<VectorizedArrayType> temp(n_entries, VectorizedArrayType(0.));

      AdvectionVelocityData<dim, Number, VectorizedArrayType> advection_data(
        this->advection, this->data);

      for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
           ++cell)
        {
          fe_eval.reinit(cell);

          advection_data.reinit(cell);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              for (auto &val : temp)
                val = 0;

              if (entries_mask[FieldVelocity] && this->advection.enabled())
                {
                  // add advection contributations
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    if (advection_data.has_velocity(ig))
                      {
                        const auto &velocity_ig = advection_data.get_velocity(
                          ig, fe_eval.quadrature_point(q));

                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            temp[ig * dim + d] = velocity_ig[d];
                          }
                      }
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

      // Write names of fields
      std::vector<std::string> names;
      if (entries_mask[FieldVelocity])
        {
          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              std::string vel_name = "vel" + std::to_string(ig);
              for (unsigned int d = 0; d < dim; ++d)
                names.push_back(vel_name);
            }
        }

      // Add data to output
      for (unsigned int c = 0; c < n_entries; ++c)
        data_out.add_data_vector(data_vectors[c], names[c]);
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
      (void)nonlinear_data;
      (void)value_result;

      for (unsigned int ig = 0; ig < this->data.n_grains(); ++ig)
        if (advection_data.has_velocity(ig))
          {
            const auto &velocity_ig =
              advection_data.get_velocity(ig, phi.quadrature_point(q));

            gradient_result[0] -= velocity_ig * phi.get_value(q)[0];
          }
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
      (void)nonlinear_data;
      (void)value_result;

      for (unsigned int ig = 0; ig < this->data.n_grains(); ++ig)
        if (advection_data.has_velocity(ig))
          {
            const auto &velocity_ig =
              advection_data.get_velocity(ig, phi.quadrature_point(q));

            gradient_result[ig] -= velocity_ig * phi.get_value(q)[0];
          }
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
      (void)nonlinear_data;
      (void)value_result;

      if (advection_data.has_velocity(igrain))
        {
          const auto &velocity_ig =
            advection_data.get_velocity(igrain, phi.quadrature_point(q));

          gradient_result -= velocity_ig * phi.get_value(q);
        }
    }

  private:
    template <int n_comp,
              int n_grains,
              int with_time_derivative,
              typename FECellIntegratorType>
    void
    do_evaluate_nonlinear_residual_cell(
      FECellIntegratorType                     &phi,
      const AlignedVector<VectorizedArrayType> &buffer) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      SinteringOperatorGenericResidualQuad<dim,
                                           VectorizedArrayType,
                                           with_time_derivative>
        quad_op(this->free_energy, this->data, buffer);

      phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                   EvaluationFlags::EvaluationFlags::gradients);

      // Reinit advection data for the current cells batch
      const AdvectionVelocityData<dim, Number, VectorizedArrayType>
        advection_data(cell, this->advection, this->data);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          auto [value_result, gradient_result] = quad_op(q, value, gradient);

          // 4) add advection contributations -> influences c AND etas
          if (this->advection.enabled())
            for (unsigned int ig = 0; ig < n_grains; ++ig)
              if (advection_data.has_velocity(ig))
                {
                  const auto &velocity_ig =
                    advection_data.get_velocity(ig, phi.quadrature_point(q));

                  gradient_result[0] -= velocity_ig * value[0];
                  gradient_result[2 + ig] -= velocity_ig * value[2 + ig];
                }


          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
      phi.integrate(EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);
    }

    template <int n_comp,
              int with_time_derivative,
              typename FECellIntegratorType>
    void
    do_evalute_history(
      FECellIntegratorType               &phi,
      AlignedVector<VectorizedArrayType> &buffer,
      const std::vector<unsigned char>   &vector_indices = {}) const
    {
      if (with_time_derivative == 2)
        {
          const auto &order         = this->data.time_data.get_order();
          const auto &weights       = this->data.time_data.get_weights();
          const auto  old_solutions = this->history.get_old_solutions();

          buffer.resize_fast(
            std::max(phi.dofs_per_cell, n_comp * phi.n_q_points));

          std::vector<const VectorType *> view;

          for (unsigned int i = 0; i < order; ++i)
            {
              if (vector_indices.size() == 0)
                phi.read_dof_values_plain(*old_solutions[i]);
              else
                {
                  view.resize(vector_indices.size());

                  for (unsigned int j = 0; j < view.size(); ++j)
                    view[j] = &old_solutions[i]->block(vector_indices[j]);

                  phi.read_dof_values_plain(view);
                }

              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                if (i == 0)
                  buffer[j] = phi.begin_dof_values()[j] * weights[i + 1];
                else
                  buffer[j] += phi.begin_dof_values()[j] * weights[i + 1];
            }

          phi.evaluate(buffer.data(), EvaluationFlags::EvaluationFlags::values);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            for (unsigned int c = 0; c < n_comp; ++c)
              buffer[q * n_comp + c] =
                phi.begin_values()[q + c * phi.n_q_points];
        }
    }

    template <int n_comp, int n_grains, int with_time_derivative>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      AssertDimension(n_comp - 2, n_grains);
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      AlignedVector<VectorizedArrayType> buffer;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);

          do_evalute_history<n_comp, with_time_derivative>(phi, buffer);

          phi.read_dof_values(src);

          do_evaluate_nonlinear_residual_cell<n_comp,
                                              n_grains,
                                              with_time_derivative>(phi,
                                                                    buffer);

          phi.distribute_local_to_global(dst);
        }
    }

    template <int with_time_derivative>
    void
    do_evaluate_nonlinear_residual_nt(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dst,
      const BlockVectorType                              &src,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      std::vector<
        std::shared_ptr<FEEvaluationData<dim, VectorizedArrayType, false>>>
        phis(this->n_grains() + 1);

      for (unsigned int i = 0; i <= this->n_grains(); ++i)
        {
          const unsigned int n_comp_nt = i + 2;
#define OPERATION(n_comp, dummy)                                 \
  (void)dummy;                                                   \
  phis[i] = std::make_shared<                                    \
    FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>>( \
    matrix_free, this->dof_index);
          EXPAND_OPERATIONS_N_COMP_NT(OPERATION);
#undef OPERATION
        }

      AlignedVector<VectorizedArrayType> buffer;
      std::vector<unsigned char>         vector_indices;
      std::vector<const VectorType *>    src_view;
      std::vector<VectorType *>          dst_view;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          // CH blocks
          vector_indices.push_back(0);
          vector_indices.push_back(1);

          // relevant AC blocks
          for (const auto b : this->data.get_relevant_grains(cell))
            vector_indices.push_back(b + 2);

          for (const auto b : vector_indices)
            {
              src_view.push_back(&src.block(b));
              dst_view.push_back(&dst.block(b));
            }

          const unsigned int n_comp_nt = src_view.size();

#define OPERATION(n_comp, dummy)                                               \
  (void)dummy;                                                                 \
  constexpr unsigned int n_grains = n_comp - 2;                                \
                                                                               \
  auto &phi =                                                                  \
    static_cast<FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &>( \
      *phis[n_grains]);                                                        \
                                                                               \
  phi.reinit(cell);                                                            \
  do_evalute_history<n_comp, with_time_derivative>(phi,                        \
                                                   buffer,                     \
                                                   vector_indices);            \
                                                                               \
  phi.read_dof_values(src_view);                                               \
                                                                               \
  do_evaluate_nonlinear_residual_cell<n_comp, n_grains, with_time_derivative>( \
    phi, buffer);                                                              \
                                                                               \
  phi.distribute_local_to_global(dst_view);
          EXPAND_OPERATIONS_N_COMP_NT(OPERATION);
#undef OPERATION

          vector_indices.clear();
          src_view.clear();
          dst_view.clear();
        }
    }

    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;
    const bool use_tensorial_mobility_gradient_on_the_fly;
  };
} // namespace Sintering
