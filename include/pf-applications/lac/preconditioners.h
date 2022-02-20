#pragma once

#include <deal.II/base/mpi_compute_index_owner_internal.h>

#include <pf-applications/base/timer.h>

namespace Preconditioners
{
  using namespace dealii;

  template <typename Number>
  class PreconditionerBase
  {
  public:
    using VectorType      = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

    virtual ~PreconditionerBase() = default;

    virtual void
    clear()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    virtual void
    do_update() = 0;
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
    clear()
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
    clear()
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
      AssertDimension(dst.n_blocks(), 1);
      AssertDimension(src.n_blocks(), 1);

      this->vmult(dst.block(0), src.block(0));
    }

    void
    do_update() override
    {
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

      std::vector<unsigned int> dummy(locally_active_dofs.n_elements());

      Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
        process(locally_owned_dofs, locally_active_dofs, comm, dummy, true);

      Utilities::MPI::ConsensusAlgorithms::Selector<
        std::pair<types::global_dof_index, types::global_dof_index>,
        unsigned int>
        consensus_algorithm;
      consensus_algorithm.run(process, comm);

      using T1 = char;
      using T2 = char;

      auto requesters = process.get_requesters();

      std::vector<std::vector<std::pair<types::global_dof_index, Number>>>
        locally_relevant_matrix_entries(locally_active_dofs.n_elements());

      dealii::Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<T1, T2>
        process_(
          [&]() {
            std::vector<unsigned int> ranks;

            for (const auto &i : requesters)
              ranks.push_back(i.first);

            return ranks;
          },
          [&](const unsigned int other_rank, std::vector<T1> &send_buffer) {
            std::vector<std::pair<
              types::global_dof_index,
              std::vector<std::pair<types::global_dof_index, Number>>>>
              temp;

            for (auto index : requesters[other_rank])
              {
                std::vector<std::pair<types::global_dof_index, Number>> t;

                for (auto entry = system_matrix_0.begin(index);
                     entry != system_matrix_0.end(index);
                     ++entry)
                  t.emplace_back(entry->column(), entry->value());

                temp.emplace_back(index, t);
              }

            send_buffer = Utilities::pack(temp, false);
          },
          [&](const unsigned int &,
              const std::vector<T1> &buffer_recv,
              std::vector<T2> &) {
            const auto temp = Utilities::unpack<std::vector<std::pair<
              types::global_dof_index,
              std::vector<std::pair<types::global_dof_index, Number>>>>>(
              buffer_recv, false);

            for (const auto &i : temp)
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
          });

      dealii::Utilities::MPI::ConsensusAlgorithms::Selector<char, char>().run(
        process_, comm);

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

    AMG(const Operator &op)
      : op(op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      precondition_amg.vmult(dst, src);
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
      precondition_amg.initialize(op.get_system_matrix(), additional_data);
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
    TrilinosWrappers::PreconditionAMG                 precondition_amg;
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
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {
      additional_data.ilu_fill = 0;
      additional_data.ilu_atol = 0.0;
      additional_data.ilu_rtol = 1.0;
      additional_data.overlap  = 0;
    }

    ~ILU()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    virtual void
    clear()
    {
      precondition_ilu.clear();
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
      VectorType dst_;                               // TOOD
      VectorType src_;                               // TOOD
      op.initialize_dof_vector(dst_);                // TOOD
      op.initialize_dof_vector(src_);                // TOOD
      VectorTools::merge_components_fast(src, src_); // TOOD

      this->vmult(dst_, src_);

      VectorTools::split_up_components_fast(dst_, dst); // TOOD
    }

    void
    do_update() override
    {
      MyScope scope(timer, "ilu::setup");
      precondition_ilu.initialize(op.get_system_matrix(), additional_data);
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionILU::AdditionalData additional_data;
    TrilinosWrappers::PreconditionILU                 precondition_ilu;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
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
    else if (label == "ILU")
      return std::make_unique<ILU<T>>(op);

    AssertThrow(false,
                ExcMessage("Preconditioner << " + label + " >> not known!"));

    return {};
  }

} // namespace Preconditioners