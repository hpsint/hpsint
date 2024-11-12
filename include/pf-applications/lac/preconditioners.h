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

#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/numerics/vector_tools.h>

namespace Preconditioners
{
  using namespace dealii;

  template <typename Number>
  class PreconditionerBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual ~PreconditionerBase() = default;

    virtual void
    clear()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual std::size_t
    memory_consumption() const
    {
      AssertThrow(false, ExcNotImplemented());

      return 0;
    }

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    virtual void
    do_update() = 0;
  };


  template <typename Operator>
  class Identity : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    Identity()
    {}

    virtual void
    clear() override
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      dst = src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      dst = src;
    }

    void
    do_update() override
    {}

    virtual std::size_t
    memory_consumption() const override
    {
      return 0;
    }
  };


  template <typename Operator>
  class InverseDiagonalMatrix
    : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    InverseDiagonalMatrix(const Operator &op)
      : op(op)
    {}

    virtual void
    clear() override
    {
      diagonal_matrix.clear();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      Assert(false, ExcNotImplemented());
      (void)dst;
      (void)src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      diagonal_matrix.vmult(dst, src);
    }

    void
    do_update() override
    {
      op.compute_inverse_diagonal(diagonal_matrix.get_vector());
    }

    virtual std::size_t
    memory_consumption() const override
    {
      return diagonal_matrix.memory_consumption();
    }

  private:
    const Operator &                op;
    DiagonalMatrix<BlockVectorType> diagonal_matrix;
  };



  template <typename Operator>
  class InverseComponentBlockDiagonalMatrix
    : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using Number          = typename Operator::value_type;
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    InverseComponentBlockDiagonalMatrix(const Operator &op)
      : op(op)
    {}

    virtual void
    clear() override
    {
      diagonal_matrix.clear();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      const unsigned int n_components =
        op.get_dof_handler().get_fe().n_components();

      dealii::Vector<Number> dst_(n_components);
      dealii::Vector<Number> src_(n_components);

      for (unsigned int cell = 0; cell < diagonal_matrix.size(); ++cell)
        {
          for (unsigned int c = 0; c < n_components; ++c)
            src_[c] = src.local_element(c + cell * n_components);

          diagonal_matrix[cell].vmult(dst_, src_);

          for (unsigned int c = 0; c < n_components; ++c)
            dst.local_element(c + cell * n_components) = dst_[c];
        }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      // MyScope scope(timer, "invsers_component_block_diagonal::vmult");

      if (src_.size() == 0 || dst_.size() == 0)
        {
          const auto partitioner = op.get_system_partitioner();

          src_.reinit(partitioner);
          dst_.reinit(partitioner);
        }

      VectorTools::merge_components_fast(src, src_); // TODO
      this->vmult(dst_, src_);
      VectorTools::split_up_components_fast(dst_, dst); // TODO
    }

    void
    do_update() override
    {
      src_.reinit(0);
      dst_.reinit(0);

      const auto &matrix = op.get_system_matrix();

      const unsigned int n_components =
        op.get_dof_handler().get_fe().n_components();
      const auto local_range = matrix.local_range();

      diagonal_matrix.resize((local_range.second - local_range.first) /
                               n_components,
                             FullMatrix<Number>(n_components));

      for (unsigned int cell = 0; cell < diagonal_matrix.size(); ++cell)
        {
          for (unsigned int i = 0; i < n_components; ++i)
            for (unsigned int j = 0; j < n_components; ++j)
              diagonal_matrix[cell][i][j] =
                matrix(i + local_range.first + cell * n_components,
                       j + local_range.first + cell * n_components);

          diagonal_matrix[cell].gauss_jordan();
        }
    }

  private:
    const Operator &                op;
    std::vector<FullMatrix<Number>> diagonal_matrix;

    mutable VectorType src_, dst_;
  };



  template <typename Operator, int dim>
  class InverseBlockDiagonalMatrix
    : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using Number          = typename VectorType::value_type;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    InverseBlockDiagonalMatrix(const Operator &op)
      : op(op)
    {}

    void
    clear() override
    {
      blocks.clear();
      weights.reinit(0);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      dst = 0.0;

      const unsigned int dofs_per_cell =
        op.get_dof_handler().get_fe().n_dofs_per_cell();

      Vector<double> vector_src(dofs_per_cell);
      Vector<double> vector_dst(dofs_per_cell);
      Vector<double> vector_weights(dofs_per_cell);

      src.update_ghost_values();

      for (const auto &cell : op.get_dof_handler().active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          // gather and ...
          cell->get_dof_values(src, vector_src);
          cell->get_dof_values(weights, vector_weights);

          // weight
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_src[i] *= vector_weights[i];

          // apply inverse element stiffness matrix
          blocks[cell->active_cell_index()].vmult(vector_dst, vector_src);

          // weight and ...
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_dst[i] *= vector_weights[i];

          // scatter
          cell->distribute_local_to_global(vector_dst, dst);
        }

      dst.compress(VectorOperation::values::add);
      src.zero_out_ghost_values();
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      AssertDimension(dst.n_blocks(), 1);
      AssertDimension(src.n_blocks(), 1);

      this->vmult(dst.block(0), src.block(0));
    }

    void
    do_update() override
    {
      const unsigned int dofs_per_cell =
        op.get_dof_handler().get_fe().n_dofs_per_cell();

      blocks.resize(op.get_dof_handler().get_triangulation().n_active_cells(),
                    FullMatrix<typename VectorType::value_type>(dofs_per_cell,
                                                                dofs_per_cell));

      compute_block_diagonal_matrix(op.get_dof_handler(),
                                    op.get_system_matrix(),
                                    blocks);
      weights = compute_weights(op.get_dof_handler());

      weights.update_ghost_values();
    }


  private:
    static void
    compute_block_diagonal_matrix(
      const DoFHandler<dim> &                                   dof_handler,
      const TrilinosWrappers::SparseMatrix &                    system_matrix_0,
      std::vector<FullMatrix<typename VectorType::value_type>> &blocks)
    {
      const unsigned int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const auto locally_owned_dofs = dof_handler.locally_owned_dofs();

      IndexSet locally_active_dofs;
      DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

      locally_active_dofs.subtract_set(locally_owned_dofs);

      const auto comm = dof_handler.get_communicator();

      const auto [dummy, requesters] =
        Utilities::MPI::compute_index_owner_and_requesters(locally_owned_dofs,
                                                           locally_active_dofs,
                                                           comm);

      using T1 = std::vector<
        std::pair<types::global_dof_index,
                  std::vector<std::pair<types::global_dof_index, Number>>>>;

      std::vector<std::vector<std::pair<types::global_dof_index, Number>>>
        locally_relevant_matrix_entries(locally_active_dofs.n_elements());


      std::vector<unsigned int> ranks;

      for (const auto &i : requesters)
        ranks.push_back(i.first);

      dealii::Utilities::MPI::ConsensusAlgorithms::selector<T1>(
        ranks,
        [&](const unsigned int other_rank) {
          T1 send_buffer;

          for (auto index : requesters.at(other_rank))
            {
              std::vector<std::pair<types::global_dof_index, Number>> t;

              for (auto entry = system_matrix_0.begin(index);
                   entry != system_matrix_0.end(index);
                   ++entry)
                t.emplace_back(entry->column(), entry->value());

              send_buffer.emplace_back(index, t);
            }

          return send_buffer;
        },
        [&](const unsigned int &, const T1 &buffer_recv) {
          for (const auto &i : buffer_recv)
            {
              auto &dst =
                locally_relevant_matrix_entries[locally_active_dofs
                                                  .index_within_set(i.first)];
              dst = i.second;
              std::sort(dst.begin(),
                        dst.end(),
                        [](const auto &a, const auto &b) {
                          return a.first < b.first;
                        });
            }
        },
        comm);

#if false
      using T = std::pair<CellId, FullMatrix<Number>>;
      std::vector<T> results;
#endif

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          cell->get_dof_indices(local_dof_indices);

          auto &cell_matrix = blocks[cell->active_cell_index()];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                if (locally_owned_dofs.is_element(
                      local_dof_indices[i])) // row is local
                  {
                    cell_matrix(i, j) = system_matrix_0(local_dof_indices[i],
                                                        local_dof_indices[j]);
                  }
                else // row is ghost
                  {
                    Assert(locally_active_dofs.is_element(local_dof_indices[i]),
                           ExcInternalError());

                    const auto &row_entries = locally_relevant_matrix_entries
                      [locally_active_dofs.index_within_set(
                        local_dof_indices[i])];

                    const auto ptr = std::lower_bound(
                      row_entries.begin(),
                      row_entries.end(),
                      std::pair<types::global_dof_index, Number>{
                        local_dof_indices[j], /*dummy*/ 0.0},
                      [](const auto a, const auto b) {
                        return a.first < b.first;
                      });

                    Assert(ptr != row_entries.end() &&
                             local_dof_indices[j] == ptr->first,
                           ExcInternalError());

                    cell_matrix(i, j) = ptr->second;
                  }
              }
#if false
          results.emplace_back(cell->id(), cell_matrix);
#endif

          cell_matrix.gauss_jordan();
        }

#if false
          const auto results_all =
            Utilities::MPI::gather(comm, Utilities::pack(results, false));

          std::vector<T> results_all_sorted;

          for(const auto & is : results_all)
            for(const auto & i : Utilities::unpack<std::vector<T>>(is, false))
              results_all_sorted.emplace_back(i);

          std::sort(results_all_sorted.begin(), results_all_sorted.end(), 
              [](const auto & a, const auto & b){return a.first < b.first;});

          if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              for(auto cell_matrix : results_all_sorted)
                {
                  cell_matrix.second.print(std::cout);
                  std::cout << std::endl;
                }
            }
          exit(0);
#endif
    }

    VectorType
    compute_weights(const DoFHandler<dim> &dof_handler_0) const
    {
      const unsigned int dofs_per_cell =
        dof_handler_0.get_fe().n_dofs_per_cell();

      LinearAlgebra::distributed::Vector<Number> weights;

      op.initialize_dof_vector(weights);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler_0.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            weights[local_dof_indices[i]] += 1;
        }

      weights.compress(VectorOperation::values::add);

      for (auto &i : weights)
        i = (i == 0.0) ? 0.0 : std::sqrt(1.0 / i);

      return weights;
    }

    const Operator &op;

    std::vector<FullMatrix<typename VectorType::value_type>> blocks;
    VectorType                                               weights;
  };



  template <typename Operator>
  class AMG : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    AMG(const Operator &op,
        const TrilinosWrappers::PreconditionAMG::AdditionalData &
          additional_data = TrilinosWrappers::PreconditionAMG::AdditionalData())
      : op(op)
      , additional_data(additional_data)
    {}

    virtual void
    clear() override
    {
      precondition_amg.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)dst;
      (void)src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "amg::vmult");

      if (src_.size() == 0 || dst_.size() == 0)
        {
          const auto partitioner = op.get_system_partitioner();

          src_.reinit(partitioner);
          dst_.reinit(partitioner);
        }

      VectorTools::merge_components_fast(src, src_); // TODO
      precondition_amg.vmult(dst_, src_);
      VectorTools::split_up_components_fast(dst_, dst); // TODO
    }

    void
    do_update() override
    {
      MyScope scope(timer, "amg::setup");

      precondition_amg.initialize(op.get_system_matrix(), additional_data);
    }

    virtual std::size_t
    memory_consumption() const override
    {
      return precondition_amg.memory_consumption() + src_.memory_consumption() +
             dst_.memory_consumption();
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    TrilinosWrappers::PreconditionAMG                 precondition_amg;

    mutable MyTimerOutput timer;

    mutable VectorType src_, dst_;
  };



  template <typename Operator>
  class BlockAMG : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    BlockAMG(
      const Operator &                                         op,
      const TrilinosWrappers::PreconditionAMG::AdditionalData &additional_data =
        TrilinosWrappers::PreconditionAMG::AdditionalData())
      : op(op)
      , additional_data(additional_data)
    {}

    virtual void
    clear() override
    {
      precondition_amg.clear();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)dst;
      (void)src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "block_amg::vmult");

      for (unsigned int b = 0; b < src.n_blocks(); ++b)
        precondition_amg[b]->vmult(dst.block(b), src.block(b));
    }

    void
    do_update() override
    {
      MyScope scope(timer, "block_amg::setup");

      const auto &block_matrix = op.get_block_system_matrix();

      precondition_amg.resize(block_matrix.size());
      for (unsigned int b = 0; b < block_matrix.size(); ++b)
        {
          precondition_amg[b] =
            std::make_shared<TrilinosWrappers::PreconditionAMG>();
          precondition_amg[b]->initialize(*block_matrix[b], additional_data);
        }
    }

    virtual std::size_t
    memory_consumption() const override
    {
      return MyMemoryConsumption::memory_consumption(precondition_amg);
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    std::vector<std::shared_ptr<TrilinosWrappers::PreconditionAMG>>
      precondition_amg;

    mutable MyTimerOutput timer;
  };



  template <typename Operator>
  class ILU : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    ILU(const Operator &op)
      : op(op)
    {
      additional_data.ilu_fill = 0;
      additional_data.ilu_atol = 0.0;
      additional_data.ilu_rtol = 1.0;
      additional_data.overlap  = 0;
    }

    ILU(
      const Operator &                                         op,
      const TrilinosWrappers::PreconditionILU::AdditionalData &additional_data)
      : op(op)
      , additional_data(additional_data)
    {}

    virtual void
    clear() override
    {
      precondition_ilu.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      MyScope scope(timer, "ilu::vmult");

      precondition_ilu.vmult(dst, src);
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "ilu::vmult");

      if (src_.size() == 0 || dst_.size() == 0)
        {
          const auto partitioner = op.get_system_partitioner();

          src_.reinit(partitioner);
          dst_.reinit(partitioner);
        }

      VectorTools::merge_components_fast(src, src_); // TODO
      precondition_ilu.vmult(dst_, src_);
      VectorTools::split_up_components_fast(dst_, dst); // TODO
    }

    void
    do_update() override
    {
      MyScope scope(timer, "ilu::setup");
      precondition_ilu.initialize(op.get_system_matrix(), additional_data);
    }

    virtual std::size_t
    memory_consumption() const override
    {
      std::size_t result = 0;

      // TODO: not implemented in deal.II
      // result += MyMemoryConsumption::memory_consumption(precondition_ilu);

      result += src_.memory_consumption();
      result += dst_.memory_consumption();

      return result;
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionILU::AdditionalData additional_data;
    TrilinosWrappers::PreconditionILU                 precondition_ilu;

    mutable MyTimerOutput timer;

    mutable VectorType src_, dst_;
  };



  template <typename Operator>
  class BlockILU : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;

    BlockILU(const Operator &op)
      : op(op)
      , single_block(op.n_unique_components() == 1)
    {
      additional_data.ilu_fill = 0;
      additional_data.ilu_atol = 0.0;
      additional_data.ilu_rtol = 1.0;
      additional_data.overlap  = 0;
    }

    BlockILU(
      const Operator &                                         op,
      const TrilinosWrappers::PreconditionILU::AdditionalData &additional_data)
      : op(op)
      , single_block(op.n_unique_components() == 1)
      , additional_data(additional_data)
    {}

    virtual void
    clear() override
    {
      precondition_ilu.clear();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)dst;
      (void)src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "block_ilu::vmult");

      AssertThrow(src.n_blocks() == dst.n_blocks(), ExcNotImplemented());

      if (single_block)
        {
          const Epetra_Operator &prec =
            precondition_ilu[0]->trilinos_operator();

          AssertThrow(dst.n_blocks() < 20, ExcNotImplemented());
          double *dst_ptrs[20], *src_ptrs[20];

          for (unsigned int b = 0; b < src.n_blocks(); ++b)
            {
              dst_ptrs[b] = dst.block(b).begin();
              src_ptrs[b] = const_cast<double *>(src.block(b).begin());
            }

          Epetra_MultiVector trilinos_dst(View,
                                          prec.OperatorRangeMap(),
                                          dst_ptrs,
                                          dst.n_blocks());
          Epetra_MultiVector trilinos_src(View,
                                          prec.OperatorDomainMap(),
                                          src_ptrs,
                                          src.n_blocks());

          const int ierr = prec.ApplyInverse(trilinos_src, trilinos_dst);
          AssertThrow(ierr == 0, ExcTrilinosError(ierr));
        }
      else
        {
          for (unsigned int b = 0; b < src.n_blocks(); ++b)
            precondition_ilu[b]->vmult(dst.block(b), src.block(b));
        }
    }

    void
    do_update() override
    {
      MyScope scope(timer, "block_ilu::setup");

      const auto &block_matrix = op.get_block_system_matrix();

      AssertThrow(single_block == false || block_matrix.size() == 1,
                  ExcNotImplemented());

      precondition_ilu.resize(block_matrix.size());
      for (unsigned int b = 0; b < block_matrix.size(); ++b)
        {
          precondition_ilu[b] =
            std::make_shared<TrilinosWrappers::PreconditionILU>();
          precondition_ilu[b]->initialize(*block_matrix[b], additional_data);
        }
    }

    virtual std::size_t
    memory_consumption() const override
    {
      std::size_t result = 0;

      // TODO: not implemented in deal.II
      // result += MyMemoryConsumption::memory_consumption(precondition_ilu);

      return result;
    }

  private:
    const Operator &op;
    const bool      single_block;

    TrilinosWrappers::PreconditionILU::AdditionalData additional_data;
    std::vector<std::shared_ptr<TrilinosWrappers::PreconditionILU>>
      precondition_ilu;

    mutable MyTimerOutput timer;
  };



  template <typename Operator>
  class GMG : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    struct PreconditionerGMGAdditionalData
    {
      double       smoothing_range               = 20;
      unsigned int smoothing_degree              = 1;
      unsigned int smoothing_eig_cg_n_iterations = 20;

      unsigned int coarse_grid_smoother_sweeps = 1;
      unsigned int coarse_grid_n_cycles        = 1;
      std::string  coarse_grid_smoother_type   = "ILU";

      unsigned int coarse_grid_maxiter = 1000;
      double       coarse_grid_abstol  = 1e-20;
      double       coarse_grid_reltol  = 1e-4;
      std::string  coarse_grid_type    = "cg_with_chebyshev";
    };

    using VectorType      = typename Operator::VectorType;
    using BlockVectorType = typename PreconditionerBase<
      typename Operator::value_type>::BlockVectorType;
    using DealiiBlockVectorType =
      LinearAlgebra::distributed::BlockVector<typename Operator::value_type>;

    using LevelMatrixType = Operator;

    using SmootherPreconditionerType = DiagonalMatrix<DealiiBlockVectorType>;
    using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                               DealiiBlockVectorType,
                                               SmootherPreconditionerType>;

    static constexpr int dim = Operator::dimension;

    using MGTransferTypeScalar = MGTransferGlobalCoarsening<dim, VectorType>;
    using MGTransferType = MGTransferBlockGlobalCoarsening<dim, VectorType>;

    GMG(const MGLevelObject<std::shared_ptr<Operator>> &op,
        const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
          &transfer)
      : op(op)
      , transfer(transfer)
    {}

    virtual void
    clear() override
    {
      preconditioner.reset();
      mg.reset();
      mg_coarse.reset();
      precondition_chebyshev.reset();
      coarse_grid_solver.reset();
      coarse_grid_solver_control.reset();
      precondition_amg.reset();
      mg_smoother.reset();
      mg_matrix.reset();
      transfer_block.reset();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)dst;
      (void)src;
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "gmg::vmult");

      Assert(preconditioner, ExcInternalError());
      preconditioner->vmult(dst, src);
    }

    void
    do_update() override
    {
      MyScope scope(timer, "gmg::setup");

      PreconditionerGMGAdditionalData additional_data;

      const unsigned int min_level = transfer->min_level();
      const unsigned int max_level = transfer->max_level();

      MGLevelObject<std::shared_ptr<Operator>> op(min_level, max_level);

      for (unsigned int l = min_level; l <= max_level; ++l)
        op[l] = this->op[l];

      // create transfer operator for block vector
      transfer_block = std::make_unique<MGTransferType>(*transfer);

      // wrap level operators
      mg_matrix = std::make_unique<mg::Matrix<DealiiBlockVectorType>>(op);

      // setup smoothers on each level
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        min_level, max_level);

      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner =
            std::make_shared<SmootherPreconditionerType>();
          op[level]->compute_inverse_diagonal(
            smoother_data[level].preconditioner->get_vector());
          smoother_data[level].smoothing_range =
            additional_data.smoothing_range;
          smoother_data[level].degree = additional_data.smoothing_degree;
          smoother_data[level].eig_cg_n_iterations =
            additional_data.smoothing_eig_cg_n_iterations;
        }

      mg_smoother =
        std::make_unique<MGSmootherPrecondition<LevelMatrixType,
                                                SmootherType,
                                                DealiiBlockVectorType>>();
      mg_smoother->initialize(op, smoother_data);

      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          DealiiBlockVectorType vec;
          op[level]->initialize_dof_vector(vec);
          mg_smoother->smoothers[level].estimate_eigenvalues(vec);
        }

      coarse_grid_solver_control =
        std::make_unique<ReductionControl>(additional_data.coarse_grid_maxiter,
                                           additional_data.coarse_grid_abstol,
                                           additional_data.coarse_grid_reltol,
                                           false,
                                           false);
      coarse_grid_solver = std::make_unique<SolverCG<DealiiBlockVectorType>>(
        *coarse_grid_solver_control);

      if (additional_data.coarse_grid_type == "cg_with_chebyshev")
        {
          typename SmootherType::AdditionalData smoother_data;

          smoother_data.preconditioner =
            std::make_shared<DiagonalMatrix<DealiiBlockVectorType>>();
          op[min_level]->compute_inverse_diagonal(
            smoother_data.preconditioner->get_vector());
          smoother_data.smoothing_range = additional_data.smoothing_range;
          smoother_data.degree          = additional_data.smoothing_degree;
          smoother_data.eig_cg_n_iterations =
            additional_data.smoothing_eig_cg_n_iterations;

          precondition_chebyshev = std::make_unique<
            PreconditionChebyshev<LevelMatrixType,
                                  DealiiBlockVectorType,
                                  DiagonalMatrix<DealiiBlockVectorType>>>();

          precondition_chebyshev->initialize(*op[min_level], smoother_data);

          mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<
            DealiiBlockVectorType,
            SolverCG<DealiiBlockVectorType>,
            LevelMatrixType,
            PreconditionChebyshev<LevelMatrixType,
                                  DealiiBlockVectorType,
                                  DiagonalMatrix<DealiiBlockVectorType>>>>(
            *coarse_grid_solver, *op[min_level], *precondition_chebyshev);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      mg = std::make_unique<Multigrid<DealiiBlockVectorType>>(*mg_matrix,
                                                              *mg_coarse,
                                                              *transfer_block,
                                                              *mg_smoother,
                                                              *mg_smoother,
                                                              min_level,
                                                              max_level);

      preconditioner = std::make_unique<
        PreconditionMG<dim, DealiiBlockVectorType, MGTransferType>>(
        dof_handler_dummy, *mg, *transfer_block);

      mg->connect_pre_smoother_step([this](const bool         flag,
                                           const unsigned int level) {
        const std::string label = "gmg::vmult::level_" + std::to_string(level);
        if (flag)
          timer.enter_subsection(label);
      });
      mg->connect_restriction([this](const bool         flag,
                                     const unsigned int level) {
        const std::string label = "gmg::vmult::level_" + std::to_string(level);
        if (flag == false)
          timer.leave_subsection(label);
      });
      mg->connect_coarse_solve([this](const bool         flag,
                                      const unsigned int level) {
        const std::string label = "gmg::vmult::level_" + std::to_string(level);
        if (flag)
          timer.enter_subsection(label);
        else
          timer.leave_subsection(label);
      });
      mg->connect_prolongation([this](const bool         flag,
                                      const unsigned int level) {
        const std::string label = "gmg::vmult::level_" + std::to_string(level);
        if (flag)
          timer.enter_subsection(label);
      });
      mg->connect_post_smoother_step([this](const bool         flag,
                                            const unsigned int level) {
        const std::string label = "gmg::vmult::level_" + std::to_string(level);
        if (flag == false)
          timer.leave_subsection(label);
      });

      preconditioner->connect_transfer_to_mg([this](const bool flag) {
        const std::string label = "gmg::vmult::transfer_to_mg";
        if (flag)
          timer.enter_subsection(label);
        else
          timer.leave_subsection(label);
      });

      preconditioner->connect_transfer_to_global([this](const bool flag) {
        const std::string label = "gmg::vmult::transfer_to_global";
        if (flag)
          timer.enter_subsection(label);
        else
          timer.leave_subsection(label);
      });
    }

  private:
    const MGLevelObject<std::shared_ptr<Operator>> &op;
    const std::shared_ptr<MGTransferTypeScalar> &   transfer;
    mutable MyTimerOutput                           timer;

    DoFHandler<dim> dof_handler_dummy;

    mutable std::unique_ptr<MGTransferType> transfer_block;

    mutable std::unique_ptr<mg::Matrix<DealiiBlockVectorType>> mg_matrix;

    mutable std::unique_ptr<MGSmootherPrecondition<LevelMatrixType,
                                                   SmootherType,
                                                   DealiiBlockVectorType>>
      mg_smoother;

    mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

    mutable std::unique_ptr<ReductionControl> coarse_grid_solver_control;
    mutable std::unique_ptr<SolverCG<DealiiBlockVectorType>> coarse_grid_solver;

    mutable std::unique_ptr<
      PreconditionChebyshev<LevelMatrixType,
                            DealiiBlockVectorType,
                            DiagonalMatrix<DealiiBlockVectorType>>>
      precondition_chebyshev;

    mutable std::unique_ptr<MGCoarseGridBase<DealiiBlockVectorType>> mg_coarse;

    mutable std::unique_ptr<Multigrid<DealiiBlockVectorType>> mg;

    mutable std::unique_ptr<
      PreconditionMG<dim, DealiiBlockVectorType, MGTransferType>>
      preconditioner;
  };


  template <typename T>
  std::unique_ptr<PreconditionerBase<typename T::value_type>>
  create(const T &op, const std::string &label)
  {
    if (label == "InverseDiagonalMatrix")
      return std::make_unique<InverseDiagonalMatrix<T>>(op);
    else if (label == "InverseComponentBlockDiagonalMatrix")
      return std::make_unique<InverseComponentBlockDiagonalMatrix<T>>(op);
    else if (label == "InverseBlockDiagonalMatrix")
      return std::make_unique<InverseBlockDiagonalMatrix<T, T::dimension>>(op);
    else if (label == "AMG")
      return std::make_unique<AMG<T>>(op);
    else if (label == "BlockAMG")
      return std::make_unique<BlockAMG<T>>(op);
    else if (label == "ILU")
      return std::make_unique<ILU<T>>(op);
    else if (label == "BlockILU")
      return std::make_unique<BlockILU<T>>(op);
    else if (label == "Identity")
      return std::make_unique<Identity<T>>();

    AssertThrow(false,
                ExcMessage("Preconditioner << " + label + " >> not known!"));

    return {};
  }

  template <typename T>
  std::unique_ptr<PreconditionerBase<typename T::value_type>>
  create(const T &                                          op,
         const std::string &                                label,
         TrilinosWrappers::PreconditionAMG::AdditionalData &additional_data)
  {
    if (label == "AMG")
      return std::make_unique<AMG<T>>(op, additional_data);
    else if (label == "BlockAMG")
      return std::make_unique<BlockAMG<T>>(op, additional_data);

    AssertThrow(
      false,
      ExcMessage(
        "Preconditioner << " + label +
        " >> not known or cannot be initialized with AMG additional data!"));

    return {};
  }

  template <typename T>
  std::unique_ptr<PreconditionerBase<typename T::value_type>>
  create(const T &                                          op,
         const std::string &                                label,
         TrilinosWrappers::PreconditionILU::AdditionalData &additional_data)
  {
    if (label == "ILU")
      return std::make_unique<ILU<T>>(op, additional_data);
    else if (label == "BlockILU")
      return std::make_unique<BlockILU<T>>(op, additional_data);

    AssertThrow(
      false,
      ExcMessage(
        "Preconditioner << " + label +
        " >> not known or cannot be initialized with ILU additional data!"));

    return {};
  }

  template <typename T>
  std::unique_ptr<PreconditionerBase<typename T::value_type>>
  create(const MGLevelObject<std::shared_ptr<T>> &op,
         const std::shared_ptr<
           MGTransferGlobalCoarsening<T::dimension, typename T::VectorType>>
           &                transfer,
         const std::string &label)
  {
    if (label == "GMG" || label == "BlockGMG")
      return std::make_unique<GMG<T>>(op, transfer);

    AssertThrow(false,
                ExcMessage("Preconditioner << " + label + " >> not known!"));

    return {};
  }

} // namespace Preconditioners
