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
      const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
      const AffineConstraints<Number> &                        constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &  data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &history,
      const bool                                               matrix_based,
      const double                                             E  = 1.0,
      const double                                             nu = 0.25,
      const Structural::MaterialPlaneType                      type =
        Structural::MaterialPlaneType::none)
      : SinteringOperatorBase<dim, Number, VectorizedArrayType, T>(matrix_free,
                                                                   constraints,
                                                                   data,
                                                                   history,
                                                                   matrix_based)
      , material(E, nu, type)
    {}

    ~SinteringOperatorCoupledBase()
    {}

    virtual unsigned int
    n_additional_components() const = 0;

    void
    update_state(const BlockVectorType &solution) override
    {
      const double c_min = 0.1;

      zero_c_constraints_indices.clear();

      const auto &partitioner = this->matrix_free.get_vector_partitioner();
      for (const auto i : partitioner->locally_owned_range())
        {
          const auto local_index = partitioner->global_to_local(i);
          if (solution.block(0).local_element(local_index) < c_min)
            zero_c_constraints_indices.emplace_back(local_index);
        }
    }

    void
    post_system_matrix_compute() const override
    {
      const auto &partitioner = this->matrix_free.get_vector_partitioner();

      for (const unsigned int index : zero_c_constraints_indices)
        for (unsigned int d = 0; d < dim; ++d)
          {
            const auto global_index = partitioner->local_to_global(index);

            const unsigned int matrix_index =
              this->n_components() * global_index + this->data.n_components() +
              n_additional_components() + d;

            this->system_matrix.clear_row(matrix_index, 1.0);
          }

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          {
            const auto global_index = partitioner->local_to_global(index);

            const unsigned int matrix_index =
              this->n_components() * global_index + this->data.n_components() +
              n_additional_components() + d;

            this->system_matrix.clear_row(matrix_index, 1.0);
          }
    }

    template <typename BlockVectorType_>
    void
    do_pre_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src_in) const
    {
      (void)dst;

      BlockVectorType_ &src = const_cast<BlockVectorType_ &>(src_in);

      zero_c_constraints_values.resize(zero_c_constraints_indices.size());

      for (unsigned int i = 0; i < zero_c_constraints_indices.size(); ++i)
        {
          const unsigned int index = zero_c_constraints_indices[i];

          for (unsigned int d = 0; d < dim; ++d)
            {
              const unsigned int block_index =
                this->data.n_components() + n_additional_components() + d;

              zero_c_constraints_values[i][d] =
                src.block(block_index).local_element(index);
              src.block(block_index).local_element(index) = 0.0;
            }
        }

      for (unsigned int d = 0; d < dim; ++d)
        displ_constraints_values[d].resize(displ_constraints_indices[d].size());

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < displ_constraints_indices[d].size(); ++i)
          {
            const auto b =
              this->data.n_components() + n_additional_components() + d;
            const auto index = displ_constraints_indices[d][i];

            displ_constraints_values[d][i] = src.block(b).local_element(index);

            src.block(b).local_element(index) = 0.0;
          }
    }

    template <typename BlockVectorType_>
    void
    do_post_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src_in) const
    {
      BlockVectorType_ &src = const_cast<BlockVectorType_ &>(src_in);

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < displ_constraints_indices[d].size(); ++i)
          {
            const auto b =
              this->data.n_components() + n_additional_components() + d;
            const auto index = displ_constraints_indices[d][i];

            src.block(b).local_element(index) = displ_constraints_values[d][i];

            dst.block(b).local_element(index) =
              src.block(b).local_element(index);
          }

      for (unsigned int i = 0; i < zero_c_constraints_indices.size(); ++i)
        {
          const auto &index = zero_c_constraints_indices[i];
          const auto &value = zero_c_constraints_values[i];

          for (unsigned int d = 0; d < dim; ++d)
            dst.block(this->data.n_components() + n_additional_components() + d)
              .local_element(index) = value[d];
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

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors_kernel(DataOut<dim> &               data_out,
                               const BlockVectorType &      vec,
                               const std::set<std::string> &fields_list) const
    {
      // Output from the parent
      SinteringOperatorBase<dim, Number, VectorizedArrayType, T>::
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
        possible_entries = {{{"vel", FieldVelocity, dim}}};

      // Get active entries to output
      const auto [entries_mask, n_entries] =
        this->get_vector_output_entries_mask(possible_entries, fields_list);

      if (n_entries == 0)
        return;

      std::vector<VectorType> data_vectors(n_entries);

      for (auto &data_vector : data_vectors)
        this->matrix_free.initialize_dof_vector(data_vector, this->dof_index);

      if (entries_mask[FieldVelocity])
        for (unsigned int d = 0; d < dim; ++d)
          {
            const auto b =
              this->data.n_components() + n_additional_components() + d;

            data_vectors[d] = vec.block(b);
            data_vectors[d] *= (this->data.time_data.get_current_dt() ?
                                  1. / this->data.time_data.get_current_dt() :
                                  0.);
          }

      // Write names of fields
      std::vector<std::string> names;
      if (entries_mask[FieldVelocity])
        {
          const std::string vel_name = "vel";
          for (unsigned int d = 0; d < dim; ++d)
            names.push_back(vel_name);
        }

      // Add data to output
      for (unsigned int c = 0; c < n_entries; ++c)
        data_out.add_data_vector(data_vectors[c], names[c]);
    }

  protected:
    void
    pre_vmult(VectorType &dst, const VectorType &src_in) const override
    {
      (void)dst;

      VectorType &src = const_cast<VectorType &>(src_in);

      zero_c_constraints_values.resize(zero_c_constraints_indices.size());

      for (unsigned int i = 0; i < zero_c_constraints_indices.size(); ++i)
        {
          const unsigned int index = zero_c_constraints_indices[i];

          for (unsigned int d = 0; d < dim; ++d)
            {
              const unsigned int matrix_index = this->n_components() * index +
                                                this->data.n_components() +
                                                n_additional_components() + d;

              zero_c_constraints_values[i][d] = src.local_element(matrix_index);
              src.local_element(matrix_index) = 0.0;
            }
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          {
            const unsigned int matrix_index = this->n_components() * index +
                                              this->data.n_components() +
                                              n_additional_components() + d;
            src.local_element(matrix_index) = 0.0;
          }
    }

    void
    post_vmult(VectorType &dst, const VectorType &src) const override
    {
      (void)src;

      for (unsigned int i = 0; i < zero_c_constraints_indices.size(); ++i)
        {
          const auto &index = zero_c_constraints_indices[i];
          const auto &value = zero_c_constraints_values[i];

          for (unsigned int d = 0; d < dim; ++d)
            {
              const unsigned int matrix_index = this->n_components() * index +
                                                n_additional_components() +
                                                this->data.n_components() + d;
              dst.local_element(matrix_index) = value[d];
            }
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          {
            const unsigned int matrix_index = this->n_components() * index +
                                              n_additional_components() +
                                              this->data.n_components() + d;
            dst.local_element(matrix_index) = 0.0;
          }
    }

    Tensor<2, dim, VectorizedArrayType>
    get_stress(const Tensor<2, dim, VectorizedArrayType> &H,
               const VectorizedArrayType &                c) const
    {
      const double c_min = 0.1;

      const auto cl = compare_and_apply_mask<SIMDComparison::less_than>(
        c, VectorizedArrayType(c_min), VectorizedArrayType(c_min), c);

      return cl * material.get_S(H);
    }

  protected:
    const Structural::StVenantKirchhoff<dim, Number, VectorizedArrayType>
      material;

    std::vector<unsigned int>                   zero_c_constraints_indices;
    mutable std::vector<Tensor<1, dim, Number>> zero_c_constraints_values;

    std::array<std::vector<unsigned int>, dim>   displ_constraints_indices;
    mutable std::array<std::vector<Number>, dim> displ_constraints_values;
  };
} // namespace Sintering
