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

#include <pf-applications/sintering/operator_base.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/tracker.h>

namespace Sintering
{
  using namespace dealii;

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
    static constexpr unsigned int n_comp_total = (dim == 3 ? 6 : 3);

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
    precompute_cell_diameters()
    {
      cell_diameters.resize(this->matrix_free.n_cell_batches());

      std::pair<unsigned int, unsigned int> range{
        0, this->matrix_free.n_cell_batches()};

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          unsigned int i = 0;

          for (; i < this->matrix_free.n_active_entries_per_cell_batch(cell);
               ++i)
            {
              const auto icell = this->matrix_free.get_cell_iterator(cell, i);
              cell_diameters[cell][i] = icell->diameter();
            }

          // Assign max double for irregular cells
          for (; i < VectorizedArrayType::size(); ++i)
            cell_diameters[cell][i] = std::numeric_limits<
              typename VectorizedArrayType::value_type>::max();
        }
    }

    bool
    check_courant(const AdvectionMechanism<dim, Number, VectorizedArrayType>
                    &          advection_mechanism,
                  const double dt) const
    {
      MyScope scope(this->timer,
                    "advection_op::check_courant",
                    this->do_timing);

      AssertThrow(cell_diameters.size() == this->matrix_free.n_cell_batches(),
                  ExcMessage(
                    "The number of precomputed cell diameters is inconsistent"
                    " with the number of current cell batches"));

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(
        this->matrix_free, this->dof_index);

      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::pair<unsigned int, unsigned int> range{
        0, this->matrix_free.n_cell_batches()};

      bool status = true;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          if (!status)
            break;

          phi.reinit(cell);

          // Reinit advection data for the current cells batch
          advection_mechanism.reinit(cell);

          const auto grain_to_relevant_grain =
            this->data.get_grain_to_relevant_grain(cell);

          for (unsigned int ig = 0; ig < n_grains() && status; ++ig)
            if (grain_to_relevant_grain[ig] !=
                  static_cast<unsigned char>(255) &&
                advection_mechanism.has_velocity(grain_to_relevant_grain[ig]))
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto &velocity_ig = advection_mechanism.get_velocity(
                    grain_to_relevant_grain[ig], phi.quadrature_point(q));

                  // Check Courant condition
                  const auto cdt = velocity_ig.norm() * dt;

                  const auto courant =
                    compare_and_apply_mask<SIMDComparison::less_than>(
                      cdt, cell_diameters[cell], zeros, ones);

                  if (courant.sum() > 0)
                    {
                      status = false;
                      break;
                    }
                }
        }

      status = static_cast<bool>(Utilities::MPI::min(
        static_cast<unsigned int>(status),
        this->matrix_free.get_dof_handler().get_communicator()));

      return status;
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

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const BlockVectorType &                               solution,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism,
      const std::pair<unsigned int, unsigned int> &         range) const
    {
      advection_mechanism.nullify_data(grain_tracker.n_segments());

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free, this->dof_index);

      FECellIntegrator<dim,
                       advection_mechanism.n_comp_volume_force_torque,
                       Number,
                       VectorizedArrayType>
        phi_ft(matrix_free, this->dof_index);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::vector<unsigned int> index_ptr = {0};
      std::vector<unsigned int> index_values;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_ft.reinit(cell);

          const auto grain_to_relevant_grain =
            this->data.get_grain_to_relevant_grain(cell);

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
                      if (grain_to_relevant_grain[ig] ==
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

                          // Disable this normalization
                          /*
                          etai_etaj      = compare_and_apply_mask<
                            SIMDComparison::greater_than>(etai_etaj,
                                                          cgb_lim,
                                                          ones,
                                                          zeros);
                          */

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

      advection_mechanism.set_grain_table(index_ptr, index_values);

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

    AlignedVector<VectorizedArrayType> cell_diameters;
  };

  class ExcCourantConditionViolated : public dealii::ExceptionBase
  {
  public:
    ExcCourantConditionViolated() = default;

    virtual ~ExcCourantConditionViolated() noexcept override = default;

    virtual void
    print_info(std::ostream &out) const override
    {
      out << message() << std::endl;
    }

    std::string
    message() const
    {
      return "The Courant condition was violated. "
             "The advection velocity is too high.";
    }
  };
} // namespace Sintering
