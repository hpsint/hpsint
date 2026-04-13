// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

#include <deal.II/matrix_free/matrix_free.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_sintering_data.h>

namespace Sintering
{
  template <int dim, typename Number, typename VectorizedArrayType>
  class CFLChecker
  {
  public:
    CFLChecker(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
               const SinteringOperatorData<dim, VectorizedArrayType> &data,
               const AdvectionMechanism<dim, Number, VectorizedArrayType>
                 &advection_mechanism)
      : matrix_free(matrix_free)
      , data(data)
      , advection_mechanism(advection_mechanism)
    {}

    void
    precompute_cell_diameters()
    {
      cell_diameters.resize(matrix_free.n_cell_batches());

      std::pair<unsigned int, unsigned int> range{0,
                                                  matrix_free.n_cell_batches()};

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          unsigned int i = 0;

          for (; i < matrix_free.n_active_entries_per_cell_batch(cell); ++i)
            {
              const auto icell        = matrix_free.get_cell_iterator(cell, i);
              cell_diameters[cell][i] = icell->diameter();
            }

          // Assign max double for irregular cells
          for (; i < VectorizedArrayType::size(); ++i)
            cell_diameters[cell][i] = std::numeric_limits<
              typename VectorizedArrayType::value_type>::max();
        }
    }

    bool
    check_courant(const double dt) const
    {
      AssertThrow(
        cell_diameters.size() == matrix_free.n_cell_batches(),
        ExcMessage(
          std::string(
            "The number of precomputed cell diameters is inconsistent") +
          std::string(" with the number of current cell batches: ") +
          std::to_string(cell_diameters.size()) + std::string(" and ") +
          std::to_string(matrix_free.n_cell_batches())));

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(matrix_free);

      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::pair<unsigned int, unsigned int> range{0,
                                                  matrix_free.n_cell_batches()};

      AdvectionVelocityData<dim, Number, VectorizedArrayType> advection_data(
        advection_mechanism, data);

      bool status = true;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          if (!status)
            break;

          phi.reinit(cell);

          // Reinit advection data for the current cells batch
          advection_data.reinit(cell);

          const auto grain_to_relevant_grain =
            data.get_grain_to_relevant_grain(cell);

          for (unsigned int ig = 0; ig < data.n_grains() && status; ++ig)
            if (advection_data.has_velocity(ig))
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  const auto &velocity_ig =
                    advection_data.get_velocity(ig, phi.quadrature_point(q));

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
        matrix_free.get_dof_handler().get_mpi_communicator()));

      return status;
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free;
    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const AdvectionMechanism<dim, Number, VectorizedArrayType>
      &advection_mechanism;

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