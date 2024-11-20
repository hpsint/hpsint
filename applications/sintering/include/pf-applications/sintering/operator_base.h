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

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <pf-applications/base/fe_integrator.h>
#include <pf-applications/base/timer.h>

#include <pf-applications/numerics/functions.h>
#include <pf-applications/numerics/vector_tools.h>

#include <pf-applications/sintering/instantiation.h>

#include <pf-applications/dofs/dof_tools.h>
#include <pf-applications/matrix_free/tools.h>

#include <fstream>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class OperatorBase : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    using QuantityCallback = std::function<
      VectorizedArrayType(const VectorizedArrayType *,
                          const Tensor<1, dim, VectorizedArrayType> *)>;

    using QuantityPredicate = std::function<VectorizedArrayType(
      const Point<dim, VectorizedArrayType> &)>;

    static const int dimension = dim;

    template <typename BlockVectorType_>
    struct check
    {
      template <typename T_>
      using do_post_vmult_t = decltype(
        std::declval<T_ const>().template do_post_vmult<BlockVectorType_>(
          std::declval<BlockVectorType_ &>(),
          std::declval<BlockVectorType_ const &>()));

      template <typename T_>
      using do_pre_vmult_t = decltype(
        std::declval<T_ const>().template do_pre_vmult<BlockVectorType_>(
          std::declval<BlockVectorType_ &>(),
          std::declval<BlockVectorType_ const &>()));
    };

    OperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const unsigned int                                  dof_index,
      const std::string                                   label        = "",
      const bool                                          matrix_based = false)
      : matrix_free(matrix_free)
      , constraints(constraints)
      , dof_index(dof_index)
      , label(label)
      , matrix_based(matrix_based)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(label != "")
      , do_timing(true)
    {}

    virtual ~OperatorBase() = default;

    virtual void
    clear()
    {
      this->system_matrix.clear();
      this->block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);

      constrained_indices.clear();
      constrained_values_src.clear();
    }

    virtual unsigned int
    n_components() const = 0;

    virtual unsigned int
    n_unique_components() const
    {
      return n_components();
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return matrix_free.get_dof_handler(dof_index);
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &
    get_matrix_free() const
    {
      return matrix_free;
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst, dof_index);
    }

    void
    initialize_dof_vector(BlockVectorType &dst) const
    {
      dst.reinit(this->n_components());
      for (unsigned int c = 0; c < this->n_components(); ++c)
        matrix_free.initialize_dof_vector(dst.block(c), dof_index);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::BlockVector<Number> &dst) const
    {
      dst.reinit(this->n_components());
      for (unsigned int c = 0; c < this->n_components(); ++c)
        matrix_free.initialize_dof_vector(dst.block(c), dof_index);
      dst.collect_sizes();
    }

    types::global_dof_index
    m() const
    {
      const auto &dof_handler = matrix_free.get_dof_handler(dof_index);

      if (dof_handler.get_fe().n_components() == 1)
        return dof_handler.n_dofs() * n_components();
      else
        return dof_handler.n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    bool
    set_timing(const bool do_timing) const
    {
      const bool old  = this->do_timing;
      this->do_timing = do_timing;
      return old;
    }

    virtual void
    vmult_internal(VectorType &dst, const VectorType &src) const
    {
#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    virtual void
    vmult_internal(BlockVectorType &dst, const BlockVectorType &src) const
    {
#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    virtual void
    vmult_internal(
      LinearAlgebra::distributed::BlockVector<Number> &      dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult_", this->do_timing);

      pre_vmult(dst, src);

      if (matrix_based == false)
        {
          if (constrained_indices.empty())
            {
              const auto &constrained_dofs =
                this->matrix_free.get_constrained_dofs(this->dof_index);

              constrained_indices.resize(constrained_dofs.size());
              for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
                constrained_indices[i] = constrained_dofs[i];
              constrained_values_src.resize(this->n_components() *
                                            constrained_indices.size());
            }

          const bool is_scalar_dof_handler =
            this->matrix_free.get_dof_handler().get_fe().n_components() == 1;
          const unsigned int user_comp =
            is_scalar_dof_handler ? n_components() : 1;

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            for (unsigned int b = 0; b < user_comp; ++b)
              {
                constrained_values_src[i * user_comp + b] =
                  src.local_element(constrained_indices[i] * user_comp + b);
                const_cast<VectorType &>(src).local_element(
                  constrained_indices[i] * user_comp + b) = 0.;
              }

          this->vmult_internal(dst, src);

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            for (unsigned int b = 0; b < user_comp; ++b)
              {
                const_cast<VectorType &>(src).local_element(
                  constrained_indices[i] * user_comp + b) =
                  constrained_values_src[i * user_comp + b];
                dst.local_element(constrained_indices[i] * user_comp + b) =
                  src.local_element(constrained_indices[i] * user_comp + b);
              }

          for (unsigned int i = 0; i < constrained_indices.size(); ++i)
            {
              const_cast<VectorType &>(src).local_element(
                constrained_indices[i]) = constrained_values_src[i];
              dst.local_element(constrained_indices[i]) =
                constrained_values_src[i];
            }
        }
      else
        {
          system_matrix.vmult(dst, src);
        }

      post_vmult(dst, src);
    }

    template <typename BlockVectorType_>
    void
    vmult(BlockVectorType_ &dst, const BlockVectorType_ &src) const
    {
      MyScope scope(this->timer, label + "::vmult", this->do_timing);

      pre_vmult(dst, src);

      if (matrix_based == false)
        {
          if (constrained_indices.empty())
            {
              const auto &constrained_dofs =
                this->matrix_free.get_constrained_dofs(this->dof_index);

              constrained_indices.resize(constrained_dofs.size());
              for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
                constrained_indices[i] = constrained_dofs[i];
              constrained_values_src.resize(this->n_components() *
                                            constrained_indices.size());
            }

          for (unsigned int b = 0; b < this->n_components(); ++b)
            for (unsigned int i = 0; i < constrained_indices.size(); ++i)
              {
                constrained_values_src[i + b * constrained_indices.size()] =
                  src.block(b).local_element(constrained_indices[i]);
                const_cast<BlockVectorType_ &>(src).block(b).local_element(
                  constrained_indices[i]) = 0.;
              }

          this->vmult_internal(dst, src);

          for (unsigned int b = 0; b < this->n_components(); ++b)
            for (unsigned int i = 0; i < constrained_indices.size(); ++i)
              {
                const_cast<BlockVectorType_ &>(src).block(b).local_element(
                  constrained_indices[i]) =
                  constrained_values_src[i + b * constrained_indices.size()];
                dst.block(b).local_element(constrained_indices[i]) =
                  src.block(b).local_element(constrained_indices[i]);
              }
        }
      else
        {
          if (src_.size() == 0 || dst_.size() == 0)
            {
              const auto partitioner = get_system_partitioner();

              src_.reinit(partitioner);
              dst_.reinit(partitioner);
            }

          VectorTools::merge_components_fast(src, src_); // TODO
          this->vmult(dst_, src_);
          VectorTools::split_up_components_fast(dst_, dst); // TODO
        }

      post_vmult(dst, src);
    }

    template <typename VectorType_>
    void
    Tvmult(VectorType_ &dst, const VectorType_ &src) const
    {
      AssertThrow(false, ExcNotImplemented());

      this->vmult(dst, src);
    }

    template <typename VectorType>
    void
    compute_inverse_diagonal(VectorType &diagonal) const
    {
      MyScope scope(this->timer, label + "::diagonal", this->do_timing);

      initialize_dof_vector(diagonal);

      Assert(internal::n_blocks(diagonal) == 1 ||
               matrix_free.get_dof_handler(dof_index).get_fe().n_components() ==
                 1,
             ExcInternalError());

#define OPERATION(c, d)                                                 \
  MatrixFreeTools::compute_diagonal(matrix_free,                        \
                                    diagonal,                           \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      for (unsigned int b = 0; b < internal::n_blocks(diagonal); ++b)
        for (auto &i : internal::block(diagonal, b))
          i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    std::shared_ptr<Utilities::MPI::Partitioner>
    get_system_partitioner() const
    {
      const auto partitioner_scalar =
        this->matrix_free.get_vector_partitioner(dof_index);

      IndexSet is(this->n_components());
      is.add_range(0, this->n_components());

      return std::make_shared<Utilities::MPI::Partitioner>(
        partitioner_scalar->locally_owned_range().tensor_product(is),
        partitioner_scalar->ghost_indices().tensor_product(is),
        partitioner_scalar->get_mpi_communicator());
    }

    TrilinosWrappers::SparseMatrix &
    get_system_matrix()
    {
      initialize_system_matrix();

      return system_matrix;
    }

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      initialize_system_matrix();

      return system_matrix;
    }

    void
    initialize_system_matrix(const bool print_stats = true) const
    {
      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer, label + "::matrix::sp", this->do_timing);

          system_matrix.clear();

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          DoFHandler<dim> dof_handler(
            this->matrix_free.get_dof_handler(dof_index).get_triangulation());
          dof_handler.distribute_dofs(
            FESystem<dim>(this->matrix_free.get_dof_handler(dof_index).get_fe(),
                          this->n_components()));

          constraints_for_matrix.clear();
          constraints_for_matrix.reinit(
            DoFTools::extract_locally_relevant_dofs(dof_handler));
          DoFTools::make_hanging_node_constraints(dof_handler,
                                                  constraints_for_matrix);
          add_matrix_constraints(dof_handler, constraints_for_matrix);
          constraints_for_matrix.close();

          dsp.reinit(dof_handler.locally_owned_dofs(),
                     dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          constraints_for_matrix,
                                          matrix_free.get_quadrature());
          dsp.compress();

          system_matrix.reinit(dsp);

          if (print_stats)
            {
              this->pcout << std::endl;
              this->pcout << "Create sparsity pattern (" << this->label
                          << ") with:" << std::endl;
              this->pcout << " - NNZ: " << system_matrix.n_nonzero_elements()
                          << std::endl;
              this->pcout << std::endl;
            }
        }

      {
        MyScope scope(this->timer,
                      label + "::matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          {
            system_matrix = 0.0; // clear existing content
          }

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    constraints_for_matrix,             \
                                    system_matrix,                      \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      {
        MyScope scope(this->timer,
                      label + "::matrix::post_compute",
                      this->do_timing);

        post_system_matrix_compute();
      }
    }

    virtual void
    add_matrix_constraints(const DoFHandler<dim> &    dof_handler,
                           AffineConstraints<Number> &matrix_constraints) const
    {
      (void)dof_handler;
      (void)matrix_constraints;
    }

    virtual void
    post_system_matrix_compute() const
    {}

    virtual void
    update_state(const BlockVectorType &solution)
    {
      (void)solution;
    }

    void
    clear_system_matrix() const
    {
      system_matrix.clear();
      block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    const TrilinosWrappers::SparsityPattern &
    get_sparsity_pattern() const
    {
      return dsp;
    }

    const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
    get_block_system_matrix() const
    {
      const bool system_matrix_is_empty = block_system_matrix.size() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer,
                        this->label + "::block_matrix::sp",
                        this->do_timing);

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          this->constraints,
                                          matrix_free.get_quadrature());
          dsp.compress();

          block_system_matrix.resize(this->n_components());
          for (unsigned int b = 0; b < this->n_components(); ++b)
            {
              block_system_matrix[b] =
                std::make_shared<TrilinosWrappers::SparseMatrix>();
              block_system_matrix[b]->reinit(dsp);
            }

          this->pcout << std::endl;
          this->pcout << "Create block sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - number of blocks: " << this->n_components()
                      << std::endl;
          this->pcout << " - NNZ:              "
                      << block_system_matrix[0]->n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      label + "::block_matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          for (unsigned int b = 0; b < this->n_components(); ++b)
            *block_system_matrix[b] = 0.0; // clear existing content

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    this->constraints,                  \
                                    block_system_matrix,                \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      return block_system_matrix;
    }

    void
    add_data_vectors(DataOut<dim> &               data_out,
                     const BlockVectorType &      vec,
                     const std::set<std::string> &fields_list) const
    {
#define OPERATION(c, d) \
  this->do_add_data_vectors<c, d>(data_out, vec, fields_list);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors(DataOut<dim> &               data_out,
                        const BlockVectorType &      vec,
                        const std::set<std::string> &fields_list) const
    {
      static_cast<const T &>(*this)
        .template do_add_data_vectors_kernel<n_comp, n_grains>(data_out,
                                                               vec,
                                                               fields_list);
    }

    virtual std::size_t
    memory_consumption() const
    {
      std::size_t result = 0;

      result += constraints_for_matrix.memory_consumption();
      result += system_matrix.memory_consumption();
      result += MyMemoryConsumption::memory_consumption(block_system_matrix);
      result += MyMemoryConsumption::memory_consumption(block_system_matrix);
      result += src_.memory_consumption();
      result += dst_.memory_consumption();
      result += MyMemoryConsumption::memory_consumption(constrained_indices);
      result += MyMemoryConsumption::memory_consumption(constrained_values_src);

      return result;
    }

    /* Compute integrals over the domain */
    std::vector<Number>
    calc_domain_quantities(std::vector<QuantityCallback> &        quantities,
                           const BlockVectorType &                vec,
                           const EvaluationFlags::EvaluationFlags eval_flags =
                             EvaluationFlags::values |
                             EvaluationFlags::gradients) const
    {
      auto predicate =
        [](const FECellIntegrator<dim, 1, Number, VectorizedArrayType> &,
           unsigned int) { return VectorizedArrayType(1.0); };

      return do_calc_domain_quantities(quantities, vec, predicate, eval_flags);
    }

    std::vector<Number>
    calc_domain_quantities(std::vector<QuantityCallback> &        quantities,
                           const BlockVectorType &                vec,
                           QuantityPredicate                      qp_predicate,
                           const EvaluationFlags::EvaluationFlags eval_flags =
                             EvaluationFlags::values |
                             EvaluationFlags::gradients) const
    {
      auto predicate =
        [&qp_predicate](
          const FECellIntegrator<dim, 1, Number, VectorizedArrayType> &fe_eval,
          unsigned int q) { return qp_predicate(fe_eval.quadrature_point(q)); };

      return do_calc_domain_quantities(quantities, vec, predicate, eval_flags);
    }

    void
    do_update()
    {
      if (this->matrix_based)
        this->initialize_system_matrix(); // assemble matrix
    }

  protected:
    template <int n_comp, int n_grains>
    void
    do_vmult_cell(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                   EvaluationFlags::EvaluationFlags::gradients);

      static_cast<const T &>(*this).template do_vmult_kernel<n_comp, n_grains>(
        phi);

      phi.integrate(EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      static_assert(n_comp > 0);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      static_assert(n_comp > 0);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      LinearAlgebra::distributed::BlockVector<Number> &      dst,
      const LinearAlgebra::distributed::BlockVector<Number> &src,
      const std::pair<unsigned int, unsigned int> &          range) const
    {
      static_assert(n_comp > 0);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    virtual void
    pre_vmult(VectorType &dst, const VectorType &src) const
    {
      (void)dst;
      (void)src;
    }

    template <typename BlockVectorType_>
    void
    pre_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src) const
    {
      (void)dst;
      (void)src;

      if constexpr (dealii::internal::is_supported_operation<
                      check<BlockVectorType_>::template do_pre_vmult_t,
                      T>)
        static_cast<const T &>(*this).template do_pre_vmult<BlockVectorType_>(
          dst, src);
    }

    virtual void
    post_vmult(VectorType &dst, const VectorType &src) const
    {
      (void)dst;
      (void)src;
    }

    template <typename BlockVectorType_>
    void
    post_vmult(BlockVectorType_ &dst, const BlockVectorType_ &src) const
    {
      (void)dst;
      (void)src;

      if constexpr (dealii::internal::is_supported_operation<
                      check<BlockVectorType_>::template do_post_vmult_t,
                      T>)
        static_cast<const T &>(*this).template do_post_vmult<BlockVectorType_>(
          dst, src);
    }

    std::vector<Number>
    do_calc_domain_quantities(
      std::vector<QuantityCallback> &        quantities,
      const BlockVectorType &                vec,
      std::function<VectorizedArrayType(
        const FECellIntegrator<dim, 1, Number, VectorizedArrayType> &,
        unsigned int)>                       predicate,
      const EvaluationFlags::EvaluationFlags eval_flags) const
    {
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> fe_eval_one(
        this->matrix_free);

      std::vector<FECellIntegrator<dim, 1, Number, VectorizedArrayType>>
        fe_eval(quantities.size(), fe_eval_one);

      std::vector<Number> q_values(quantities.size());

      vec.update_ghost_values();

      const unsigned n_quadrature_points =
        this->matrix_free.get_quadrature().size();

      Table<2, VectorizedArrayType> buffer_values(n_quadrature_points,
                                                  this->n_components());
      Table<2, Tensor<1, dim, VectorizedArrayType>> buffer_gradients(
        n_quadrature_points, this->n_components());

      for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
           ++cell)
        {
          fe_eval_one.reinit(cell);

          for (unsigned int i = 0; i < quantities.size(); ++i)
            fe_eval[i].reinit(cell);

          for (unsigned int c = 0; c < this->n_components(); ++c)
            {
              fe_eval_one.read_dof_values_plain(vec.block(c));
              fe_eval_one.evaluate(eval_flags);

              for (unsigned int q = 0; q < fe_eval_one.n_q_points; ++q)
                {
                  if (eval_flags & EvaluationFlags::values)
                    buffer_values(q, c) = fe_eval_one.get_value(q);

                  if (eval_flags & EvaluationFlags::gradients)
                    buffer_gradients(q, c) = fe_eval_one.get_gradient(q);
                }
            }

          for (unsigned int q = 0; q < fe_eval_one.n_q_points; ++q)
            {
              // Take the first FECellIntegrator since all of them have the same
              // quadrature points
              const auto filter = predicate(fe_eval_one, q);

              const auto &val  = buffer_values[q];
              const auto &grad = buffer_gradients[q];

              for (unsigned int i = 0; i < quantities.size(); ++i)
                {
                  const auto &q_eval       = quantities[i];
                  const auto  value_result = q_eval(&val[0], &grad[0]) * filter;

                  fe_eval[i].submit_value(value_result, q);
                }
            }

          for (unsigned int i = 0; i < quantities.size(); ++i)
            {
              const auto vals = fe_eval[i].integrate_value();
              q_values[i] +=
                std::accumulate(vals.begin(), vals.end(), Number(0));
            }
        }

      vec.zero_out_ghost_values();

      for (unsigned int i = 0; i < quantities.size(); ++i)
        q_values[i] =
          Utilities::MPI::sum<Number>(q_values[i],
                                      this->matrix_free
                                        .get_dof_handler(this->dof_index)
                                        .get_communicator());

      return q_values;
    }

    template <auto n_data_variants, typename O>
    auto
    get_vector_output_entries_mask(
      const std::array<std::tuple<std::string, O, unsigned int>,
                       n_data_variants> &possible_entries,
      const std::set<std::string> &      fields_list) const
    {
      // A better design is possible, but at the moment this is sufficient
      std::array<bool, n_data_variants> entries_mask;
      entries_mask.fill(false);

      unsigned int n_entries = 0;

      for (unsigned int i = 0; i < possible_entries.size(); i++)
        {
          const auto &entry = possible_entries[i];
          if (fields_list.count(std::get<0>(entry)))
            {
              entries_mask[std::get<1>(entry)] = true;
              n_entries += std::get<2>(entry);
            }
        }

      return std::make_tuple(entries_mask, n_entries);
    }

  protected:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const AffineConstraints<Number> &                   constraints;
    mutable AffineConstraints<Number>                   constraints_for_matrix;

    const unsigned int dof_index;

    const std::string label;
    const bool        matrix_based;

    mutable TrilinosWrappers::SparsityPattern dsp;
    mutable TrilinosWrappers::SparseMatrix    system_matrix;

    mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>>
      block_system_matrix;

    ConditionalOStream    pcout;
    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    mutable VectorType src_, dst_;

    mutable std::vector<unsigned int> constrained_indices;
    mutable std::vector<Number>       constrained_values_src;
  };

} // namespace Sintering
