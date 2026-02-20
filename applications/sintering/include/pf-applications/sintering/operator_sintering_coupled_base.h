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

#include <pf-applications/sintering/operator_sintering_base.h>

#include <pf-applications/structural/stvenantkirchhoff.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class SinteringOperatorCoupledBase
    : public SinteringOperatorBase<dim, Number, VectorizedArrayType, T>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorCoupledBase(
      const MatrixFree<dim, Number, VectorizedArrayType>      &matrix_free,
      const AffineConstraints<Number>                         &constraints,
      const FreeEnergy<VectorizedArrayType>                   &free_energy,
      const SinteringOperatorData<dim, VectorizedArrayType>   &data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &history,
      const bool                                               matrix_based,
      const double                                             E  = 1.0,
      const double                                             nu = 0.25,
      const Structural::MaterialPlaneType                      type =
        Structural::MaterialPlaneType::none,
      const double c_min = 0.1)
      : SinteringOperatorBase<dim, Number, VectorizedArrayType, T>(matrix_free,
                                                                   constraints,
                                                                   free_energy,
                                                                   data,
                                                                   history,
                                                                   matrix_based)
      , material(E, nu, type)
      , c_min(c_min)
    {}

    ~SinteringOperatorCoupledBase()
    {}

    virtual unsigned int
    n_additional_components() const = 0;

    void
    update_state(const BlockVectorType &solution) override
    {
      const double c_zero = 0.1;

      zero_c_constraints_indices.clear();

      const auto &partitioner = this->matrix_free.get_vector_partitioner();
      for (const auto i : partitioner->locally_owned_range())
        {
          const auto local_index = partitioner->global_to_local(i);
          if (solution.block(0).local_element(local_index) < c_zero)
            zero_c_constraints_indices.emplace_back(local_index);
        }
    }

    std::array<std::vector<unsigned int>, dim> &
    get_zero_constraints_indices()
    {
      return displ_constraints_indices;
    }

    const std::array<std::vector<unsigned int>, dim> &
    get_zero_constraints_indices() const
    {
      return displ_constraints_indices;
    }

  protected:
    Tensor<2, dim, VectorizedArrayType>
    get_stress(const Tensor<2, dim, VectorizedArrayType> &H,
               const VectorizedArrayType                 &c) const
    {
      const auto cl = compare_and_apply_mask<SIMDComparison::less_than>(
        c, VectorizedArrayType(c_min), VectorizedArrayType(c_min), c);

      return cl * material.get_S(H);
    }

  protected:
    const Structural::StVenantKirchhoff<dim, Number, VectorizedArrayType>
      material;

    const double c_min;

    std::vector<unsigned int>                   zero_c_constraints_indices;
    mutable std::vector<Tensor<1, dim, Number>> zero_c_constraints_values;

    std::array<std::vector<unsigned int>, dim>   displ_constraints_indices;
    mutable std::array<std::vector<Number>, dim> displ_constraints_values;
  };
} // namespace Sintering
