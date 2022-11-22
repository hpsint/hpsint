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
      const double                                             nu = 0.25)
      : SinteringOperatorBase<dim, Number, VectorizedArrayType, T>(matrix_free,
                                                                   constraints,
                                                                   data,
                                                                   history,
                                                                   matrix_based)
      , material(E, nu, Structural::TWO_DIM_TYPE::PLAIN_STRAIN)
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
          if (solution.block(0)[local_index] < c_min)
            zero_c_constraints_indices.emplace_back(local_index);
        }
    }

    void
    post_system_matrix_compute() const override
    {
      for (const unsigned int index : zero_c_constraints_indices)
        for (unsigned int d = 0; d < dim; ++d)
          {
            const unsigned int matrix_index = this->n_components() * index +
                                              this->data.n_components() +
                                              n_additional_components() + d;

            this->system_matrix.clear_row(matrix_index, 1.0);
          }

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          {
            const unsigned int matrix_index = this->n_components() * index +
                                              this->data.n_components() +
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
        for (const unsigned int index : displ_constraints_indices[d])
          src.block(this->data.n_components() + n_additional_components() + d)
            .local_element(index) = 0.0;
    }

    template <typename BlockVectorType_>
    void
    do_post_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src) const
    {
      (void)src;

      for (unsigned int i = 0; i < zero_c_constraints_indices.size(); ++i)
        {
          const auto &index = zero_c_constraints_indices[i];
          const auto &value = zero_c_constraints_values[i];

          for (unsigned int d = 0; d < dim; ++d)
            dst.block(this->data.n_components() + n_additional_components() + d)
              .local_element(index) = value[d];
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          dst.block(this->data.n_components() + n_additional_components() + d)
            .local_element(index) = 0.0;
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

  protected:
    const Structural::StVenantKirchhoff<dim, Number, VectorizedArrayType>
      material;

    std::vector<unsigned int>           zero_c_constraints_indices;
    mutable std::vector<Tensor<1, dim>> zero_c_constraints_values;

    std::array<std::vector<unsigned int>, dim> displ_constraints_indices;
  };
} // namespace Sintering
