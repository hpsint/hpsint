// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2026 by the hpsint authors
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

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/instantiation.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/matrix_free/tools.h>

namespace Sintering
{
  using namespace dealii;

  template <typename Number>
  class AdvectionOperator
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual ~AdvectionOperator() = default;

    virtual void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const = 0;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionOperatorGeneric : public AdvectionOperator<Number>
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type = Number;

    using T = AdvectionOperatorGeneric<dim, Number, VectorizedArrayType>;

    // Force, torque and grain volume
    static constexpr unsigned int n_comp_total = (dim == 3 ? 6 : 3);

    AdvectionOperatorGeneric(
      const double                                           k,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(k)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    ~AdvectionOperatorGeneric() override
    {}

    void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      MyScope scope(timer, "advection_op_gen::evaluate_forces", do_timing);

      advection_mechanism.nullify_data(grain_tracker.n_segments());

      // We do not have an output vector
      BlockVectorType dummy(1);

#define OPERATION(c, d)                                  \
  MyMatrixFreeTools::cell_loop_wrapper(                  \
    matrix_free,                                         \
    &AdvectionOperatorGeneric::do_evaluate_forces<c, d>, \
    this,                                                \
    dummy,                                               \
    src,                                                 \
    pre_operation,                                       \
    post_operation);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    advection_mechanism.get_grains_data().data(),
                    advection_mechanism.get_grains_data().size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);
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
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dummy,
      const BlockVectorType                              &solution,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      (void)dummy;

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free);

      FECellIntegrator<dim,
                       AdvectionMechanism<dim, Number, VectorizedArrayType>::
                         n_comp_volume_force_torque,
                       Number,
                       VectorizedArrayType>
        phi_ft(matrix_free);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::vector<unsigned int> &index_ptr =
        advection_mechanism.get_index_ptr();
      std::vector<unsigned int> &index_values =
        advection_mechanism.get_index_values();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_ft.reinit(cell);

          const auto grain_to_relevant_grain =
            data.get_grain_to_relevant_grain(cell);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              if (grain_to_relevant_grain[ig] ==
                  static_cast<unsigned char>(255))
                continue;

              Point<dim, VectorizedArrayType> rc;

              std::vector<std::pair<unsigned int, unsigned int>> segments(
                matrix_free.n_active_entries_per_cell_batch(cell));

              unsigned int i = 0;

              for (; i < segments.size(); ++i)
                {
                  bool set_invalid = false;

                  const auto icell = matrix_free.get_cell_iterator(cell, i);
                  const auto cell_index = icell->global_active_cell_index();

                  const unsigned int particle_id =
                    grain_tracker.get_particle_index(ig, cell_index);

                  if (particle_id != numbers::invalid_unsigned_int)
                    {
                      const auto grain_and_segment =
                        grain_tracker.get_grain_and_segment(ig, particle_id);

                      if (grain_and_segment.first !=
                          numbers::invalid_unsigned_int)
                        {
                          const auto &rc_i = grain_tracker.get_segment_center(
                            grain_and_segment.first, grain_and_segment.second);

                          for (unsigned int d = 0; d < dim; ++d)
                            rc[d][i] = rc_i[d];

                          segments[i] = grain_and_segment;

                          index_values.push_back(
                            grain_tracker.get_grain_segment_index(
                              grain_and_segment.first,
                              grain_and_segment.second));
                        }
                      else
                        {
                          set_invalid = true;
                        }
                    }
                  else
                    {
                      set_invalid = true;
                    }

                  if (set_invalid)
                    {
                      segments[i] =
                        std::make_pair(numbers::invalid_unsigned_int,
                                       numbers::invalid_unsigned_int);

                      index_values.push_back(numbers::invalid_unsigned_int);
                    }
                }

              for (; i < VectorizedArrayType::size(); ++i)
                index_values.push_back(numbers::invalid_unsigned_int);

              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                {
                  const auto val  = phi_sint.get_value(q);
                  const auto grad = phi_sint.get_gradient(q);

                  auto &c          = val[0];
                  auto &eta_i      = val[2 + ig];
                  auto &eta_grad_i = grad[2 + ig];

                  const auto &r = phi_sint.quadrature_point(q);

                  Tensor<1,
                         AdvectionMechanism<dim, Number, VectorizedArrayType>::
                           n_comp_volume_force_torque,
                         VectorizedArrayType>
                                                      value_result;
                  Tensor<1, dim, VectorizedArrayType> force;
                  moment_t<dim, VectorizedArrayType>  torque;
                  torque = 0;

                  // Compute force and torque acting on grain i from each of the
                  // other grains
                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      if (grain_to_relevant_grain[jg] ==
                          static_cast<unsigned char>(255))
                        continue;

                      if (ig != jg)
                        {
                          auto &eta_j      = val[2 + jg];
                          auto &eta_grad_j = grad[2 + jg];

                          // Vector normal to the grain boundary
                          Tensor<1, dim, VectorizedArrayType> dF =
                            eta_grad_i - eta_grad_j;

                          // Filter to detect grain boundary
                          auto etai_etaj = eta_i * eta_j;

                          // Normalize or not
                          if (cgb > 0)
                            {
                              if (smoothening > 0)
                                etai_etaj =
                                  ones /
                                  (ones + std::exp(-smoothening *
                                                   (etai_etaj - cgb_lim)));
                              else
                                etai_etaj = compare_and_apply_mask<
                                  SIMDComparison::greater_than>(etai_etaj,
                                                                cgb_lim,
                                                                ones,
                                                                zeros);
                            }
                          else
                            {
                              etai_etaj *= 4.;
                            }

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
                  if constexpr (moment_s<dim, VectorizedArrayType> == 1)
                    value_result[dim + 1] = torque;
                  else
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
                    {
                      const auto segment_index =
                        grain_tracker.get_grain_segment_index(
                          grain_and_segment.first, grain_and_segment.second);

                      const auto &rc_i = grain_tracker.get_segment_center(
                        grain_and_segment.first, grain_and_segment.second);

                      for (unsigned int d = 0; d < dim; ++d)
                        advection_mechanism.grain_center(segment_index)[d] +=
                          rc_i[d];

                      for (unsigned int d = 0;
                           d < advection_mechanism.n_comp_volume_force_torque;
                           ++d)
                        advection_mechanism.grain_data(segment_index)[d] +=
                          volume_force_torque[d][i];
                    }
                }
            }

          AssertDimension(index_values.size() % VectorizedArrayType::size(), 0);

          index_ptr.push_back(index_values.size());
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    const double k;
    const double cgb;
    const double ceq;
    const double smoothening;

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const GrainTracker::Tracker<dim, Number>              &grain_tracker;

    AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionOperatorWeighted : public AdvectionOperator<Number>
  {
  public:
    using T = AdvectionOperatorWeighted<dim, Number, VectorizedArrayType>;

    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type = Number;

    // Some required number of components
    static constexpr unsigned int n_comp_volume   = 1;
    static constexpr unsigned int n_comp_force_gb = dim + 1;
    static constexpr unsigned int n_comp_total    = n_comp_force_gb;

    AdvectionOperatorWeighted(
      const double                                           k,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(k)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    ~AdvectionOperatorWeighted() override
    {}

    void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      MyScope scope(timer, "advection_op_wgt::evaluate_forces", do_timing);

      const auto n_segments = grain_tracker.n_segments();

      advection_mechanism.nullify_data(n_segments);

      // We do not have an output vector
      BlockVectorType dummy(1);

      // TODO: optimize - use sparsity pattern based on particle interactions
      grains_data_forces_gbs.assign(n_segments * n_comp_force_gb * n_segments,
                                    0.);
      grains_data_volumes.assign(n_segments, 0.);

#define OPERATION(c, d)                                   \
  MyMatrixFreeTools::cell_loop_wrapper(                   \
    matrix_free,                                          \
    &AdvectionOperatorWeighted::do_evaluate_forces<c, d>, \
    this,                                                 \
    dummy,                                                \
    src,                                                  \
    pre_operation,                                        \
    post_operation);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    grains_data_forces_gbs.data(),
                    grains_data_forces_gbs.size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    grains_data_volumes.data(),
                    grains_data_volumes.size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      // Compute forces and communicate them to the regular advection
      advection_mechanism.nullify_data(grain_tracker.n_segments());
      for (unsigned int i = 0; i < n_segments; ++i)
        {
          const auto it =
            grains_data_forces_gbs.begin() + i * (n_comp_force_gb * n_segments);

          Tensor<1, dim, Number> force_i;
          for (unsigned int j = 0; j < n_segments; ++j)
            {
              const auto offset_ij = it + j * n_comp_force_gb;
              const auto gb_ij     = *(offset_ij + dim);

              if (i != j && std::abs(gb_ij) > 1e-10)
                {
                  Tensor<1, dim, Number> force_ij(
                    make_array_view(offset_ij, offset_ij + dim));

                  force_ij *= 1. / gb_ij;
                  force_i += force_ij;
                }
            }

          advection_mechanism.grain_data(i)[0] = grains_data_volumes[i];

          for (unsigned int d = 0; d < dim; ++d)
            advection_mechanism.grain_data(i)[d + 1] = force_i[d];

          for (unsigned int d = 1 + dim;
               d < AdvectionMechanism<dim, Number, VectorizedArrayType>::
                     n_comp_volume_force_torque;
               ++d)
            advection_mechanism.grain_data(i)[d] = 0.;
        }
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
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dummy,
      const BlockVectorType                              &solution,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      (void)dummy;

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free);

      FECellIntegrator<dim, n_comp_volume, Number, VectorizedArrayType> phi_v(
        matrix_free);

      FECellIntegrator<dim, n_comp_force_gb, Number, VectorizedArrayType>
        phi_fg(matrix_free);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::vector<unsigned int> &index_ptr =
        advection_mechanism.get_index_ptr();
      std::vector<unsigned int> &index_values =
        advection_mechanism.get_index_values();

      auto init_segments = [&, this](const auto cell, const unsigned int ig) {
        std::vector<std::pair<unsigned int, unsigned int>> segments(
          matrix_free.n_active_entries_per_cell_batch(cell));

        unsigned int i = 0;

        for (; i < segments.size(); ++i)
          {
            bool set_invalid = false;

            const auto icell      = matrix_free.get_cell_iterator(cell, i);
            const auto cell_index = icell->global_active_cell_index();

            const unsigned int particle_id =
              grain_tracker.get_particle_index(ig, cell_index);

            if (particle_id != numbers::invalid_unsigned_int)
              {
                const auto grain_and_segment =
                  grain_tracker.get_grain_and_segment(ig, particle_id);

                if (grain_and_segment.first != numbers::invalid_unsigned_int)
                  {
                    segments[i] = grain_and_segment;

                    index_values.push_back(
                      grain_tracker.get_grain_segment_index(
                        grain_and_segment.first, grain_and_segment.second));
                  }
                else
                  {
                    set_invalid = true;
                  }
              }
            else
              {
                set_invalid = true;
              }

            if (set_invalid)
              {
                segments[i] = std::make_pair(numbers::invalid_unsigned_int,
                                             numbers::invalid_unsigned_int);

                index_values.push_back(numbers::invalid_unsigned_int);
              }
          }

        for (; i < VectorizedArrayType::size(); ++i)
          index_values.push_back(numbers::invalid_unsigned_int);

        return segments;
      };

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_v.reinit(cell);
          phi_fg.reinit(cell);

          const auto grain_to_relevant_grain =
            data.get_grain_to_relevant_grain(cell);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              if (grain_to_relevant_grain[ig] ==
                  static_cast<unsigned char>(255))
                continue;

              const auto segments_i = init_segments(cell, ig);

              // Deal with volume only first
              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                {
                  const auto val = phi_sint.get_value(q);

                  auto &eta_i = val[2 + ig];

                  Tensor<1, 1, VectorizedArrayType> value_result;
                  value_result[0] = eta_i;

                  phi_v.submit_value(value_result, q);
                }

              const auto volume = phi_v.integrate_value();

              for (unsigned int k = 0; k < segments_i.size(); ++k)
                {
                  const auto &grain_and_segment_i = segments_i[k];

                  if (grain_and_segment_i.first ==
                      numbers::invalid_unsigned_int)
                    continue;

                  // i index
                  const auto segment_index_i =
                    grain_tracker.get_grain_segment_index(
                      grain_and_segment_i.first, grain_and_segment_i.second);

                  // store volume
                  grains_data_volumes[segment_index_i] += volume[k];
                }

              // Now deal with forces
              for (unsigned int jg = 0; jg < n_grains; ++jg)
                {
                  if (ig == jg)
                    continue;

                  if (grain_to_relevant_grain[jg] ==
                      static_cast<unsigned char>(255))
                    continue;

                  const auto segments_j = init_segments(cell, jg);

                  for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                    {
                      const auto val  = phi_sint.get_value(q);
                      const auto grad = phi_sint.get_gradient(q);

                      auto &c          = val[0];
                      auto &eta_i      = val[2 + ig];
                      auto &eta_grad_i = grad[2 + ig];

                      auto &eta_j      = val[2 + jg];
                      auto &eta_grad_j = grad[2 + jg];

                      Tensor<1, n_comp_force_gb, VectorizedArrayType>
                        value_result;

                      // Vector normal to the grain boundary
                      Tensor<1, dim, VectorizedArrayType> dF =
                        eta_grad_i - eta_grad_j;

                      // Filter to detect grain boundary
                      auto etai_etaj = eta_i * eta_j;

                      // Normalize or not
                      if (cgb > 0)
                        {
                          if (smoothening > 0)
                            etai_etaj =
                              ones / (ones + std::exp(-smoothening *
                                                      (etai_etaj - cgb_lim)));
                          else
                            etai_etaj = compare_and_apply_mask<
                              SIMDComparison::greater_than>(etai_etaj,
                                                            cgb_lim,
                                                            ones,
                                                            zeros);
                        }
                      else
                        {
                          etai_etaj *= 4.;
                        }

                      // Compute force component per cell
                      dF *= k * (c - ceq) * etai_etaj;

                      VectorizedArrayType gb_area = eta_i * eta_j;

                      // Force acting between grains i and j
                      for (unsigned int d = 0; d < dim; ++d)
                        value_result[d] = dF[d];

                      // Store GB area
                      value_result[dim] = gb_area;

                      phi_fg.submit_value(value_result, q);
                    }

                  const auto force_gb = phi_fg.integrate_value();

                  for (unsigned int k = 0; k < segments_i.size(); ++k)
                    {
                      const auto &grain_and_segment_i = segments_i[k];

                      if (grain_and_segment_i.first ==
                          numbers::invalid_unsigned_int)
                        continue;

                      const auto &grain_and_segment_j = segments_j[k];

                      if (grain_and_segment_j.first ==
                          numbers::invalid_unsigned_int)
                        continue;

                      // i index
                      const auto segment_index_i =
                        grain_tracker.get_grain_segment_index(
                          grain_and_segment_i.first,
                          grain_and_segment_i.second);

                      const auto offset_i = grain_tracker.n_segments() *
                                            n_comp_force_gb * segment_index_i;

                      // j index
                      const auto segment_index_j =
                        grain_tracker.get_grain_segment_index(
                          grain_and_segment_j.first,
                          grain_and_segment_j.second);

                      for (unsigned int d = 0; d < n_comp_force_gb; ++d)
                        grains_data_forces_gbs
                          [offset_i + n_comp_force_gb * segment_index_j + d] +=
                          force_gb[d][k];
                    }
                }
            }

          AssertDimension(index_values.size() % VectorizedArrayType::size(), 0);

          index_ptr.push_back(index_values.size());
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    const double k;
    const double cgb;
    const double ceq;
    const double smoothening;

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const GrainTracker::Tracker<dim, Number>              &grain_tracker;

    AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism;

    mutable std::vector<Number> grains_data_forces_gbs;
    mutable std::vector<Number> grains_data_volumes;
  };
} // namespace Sintering
