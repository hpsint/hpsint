#pragma once

#include <pf-applications/sintering/operator_base.h>

#include <pf-applications/structural/stvenantkirchhoff.h>
#include <pf-applications/structural/tools.h>

namespace Structural
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim, typename Number, typename VectorizedArrayType>
  class LinearElasticOperator
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        LinearElasticOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = LinearElasticOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    using ConstraintsCallback =
      std::function<void(const DoFHandler<dim> &dof_handler,
                         AffineConstraints<Number> &)>;

    using ExternalLoadingCallback =
      std::function<Tensor<1, dim, VectorizedArrayType>(
        FECellIntegrator<dim, dim, Number, VectorizedArrayType> &phi,
        const unsigned int                                       q)>;

    LinearElasticOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const bool                                          matrix_based,
      const double                                        E,
      const double                                        nu,
      const Structural::MaterialPlaneType                 plane_type =
        Structural::MaterialPlaneType::none,
      ConstraintsCallback     imposition = {},
      ExternalLoadingCallback loading    = {})
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     LinearElasticOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "elastic_linear_op",
          matrix_based)
      , material(E, nu, plane_type)
      , constraints_imposition(imposition)
      , external_loading(loading)
    {}

    ~LinearElasticOperator()
    {}

    void
    evaluate_nonlinear_residual(BlockVectorType &      dst,
                                const BlockVectorType &src) const
    {
      MyScope scope(this->timer,
                    "elastic_linear_op::nonlinear_residual",
                    this->do_timing);

      dst = 0.0;

      src.update_ghost_values();

      do_evaluate_nonlinear_residual(this->matrix_free,
                                     dst,
                                     src,
                                     std::pair<unsigned int, unsigned int>{
                                       0, this->matrix_free.n_cell_batches()});

      src.zero_out_ghost_values();
      dst.compress(VectorOperation::add);

      // Apply manual constraints
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          dst.block(d).local_element(dirichlet_constraints_indices[d][i]) =
            dirichlet_constraints_values[d][i];
    }

    void
    do_update()
    {
      if (this->matrix_based)
        this->initialize_system_matrix(); // assemble matrix
    }

    unsigned int
    n_components() const override
    {
      return dim;
    }

    unsigned int
    n_grains() const
    {
      return dim;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      (void)n_grains;
      return dim;
    }

    void
    add_matrix_constraints(
      const DoFHandler<dim> &    dof_handler,
      AffineConstraints<Number> &matrix_constraints) const override
    {
      if (constraints_imposition)
        constraints_imposition(dof_handler, matrix_constraints);
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      typename FECellIntegratorType::value_type zero_result;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto grad = phi.get_gradient(q);
          const auto S    = material.get_S(grad);

          phi.submit_gradient(S, q);
          phi.submit_value(zero_result, q);
        }
    }

    template <typename BlockVectorType_>
    void
    do_pre_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src_in) const
    {
      (void)dst;

      BlockVectorType_ &src = const_cast<BlockVectorType_ &>(src_in);

      // apply inhomogeneous DBC on source vector
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          src.block(d).local_element(dirichlet_constraints_indices[d][i]) =
            dirichlet_constraints_values[d][i];
    }

    template <typename BlockVectorType_>
    void
    do_post_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src_in) const
    {
      BlockVectorType_ &src = const_cast<BlockVectorType_ &>(src_in);

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          {
            src.block(d).local_element(dirichlet_constraints_indices[d][i]) =
              0.0;
            dst.block(d).local_element(dirichlet_constraints_indices[d][i]) =
              dirichlet_constraints_values[d][i];
          }
    }

    void
    post_system_matrix_compute() const override
    {
      const auto &partitioner = this->matrix_free.get_vector_partitioner();

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          {
            const auto global_index =
              partitioner->local_to_global(dirichlet_constraints_indices[d][i]);

            const unsigned int matrix_index = dim * global_index + d;

            this->system_matrix.clear_row(matrix_index, 1.0);
          }
    }

    void
    attach_dirichlet_boundary_conditions(const AffineConstraints<Number> &ac,
                                         const unsigned int               d)
    {
      AssertIndexRange(d, dim);

      // clear old contents
      dirichlet_constraints_indices[d].clear();
      dirichlet_constraints_values[d].clear();

      // loop over locally-owned indices and collect constrained indices and
      // possibly the corresponding inhomogenity
      const auto &partitioner = this->matrix_free.get_vector_partitioner();
      for (const auto i : partitioner->locally_owned_range())
        {
          if (ac.is_constrained(i))
            {
              Number value = 0.0;

              if (ac.is_inhomogeneously_constrained(i))
                value = ac.get_inhomogeneity(i);

              dirichlet_constraints_indices[d].emplace_back(
                partitioner->global_to_local(i));
              dirichlet_constraints_values[d].emplace_back(value);
            }
        }
    }

  protected:
    void
    pre_vmult(VectorType &dst, const VectorType &src_in) const override
    {
      (void)dst;

      VectorType &src = const_cast<VectorType &>(src_in);

      // apply inhomogeneous DBC on source vector
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          {
            const unsigned int matrix_index =
              dim * dirichlet_constraints_indices[d][i] + d;
            src.local_element(matrix_index) =
              dirichlet_constraints_values[d][i];
          }
    }

    void
    post_vmult(VectorType &dst, const VectorType &src_in) const override
    {
      VectorType &src = const_cast<VectorType &>(src_in);

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < dirichlet_constraints_indices[d].size();
             ++i)
          {
            const unsigned int matrix_index =
              dim * dirichlet_constraints_indices[d][i] + d;
            src.local_element(matrix_index) = 0.0;
            dst.local_element(matrix_index) =
              dirichlet_constraints_values[d][i];
          }
    }

  private:
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, dim, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::values |
                                EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto grad = phi.get_gradient(q);

              const auto S = material.get_S(grad);

              phi.submit_gradient(S, q);

              // External loading
              if (external_loading)
                {
                  auto applied_force = external_loading(phi, q);
                  phi.submit_value(applied_force, q);
                }
            }
          phi.integrate_scatter(EvaluationFlags::values |
                                  EvaluationFlags::gradients,
                                dst);
        }
    }

    const StVenantKirchhoff<dim, Number, VectorizedArrayType> material;

    const ConstraintsCallback     constraints_imposition;
    const ExternalLoadingCallback external_loading;

    std::array<std::vector<unsigned int>, dim> dirichlet_constraints_indices;
    std::array<std::vector<Number>, dim>       dirichlet_constraints_values;
  };
} // namespace Structural
