#pragma once

#include <pf-applications/timer.h>

namespace Preconditioners
{
  using namespace dealii;

  template <typename Number>
  class PreconditionerBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    virtual ~PreconditionerBase() = default;

    virtual void
    clear()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    do_update() = 0;
  };



  template <typename Operator>
  class InverseDiagonalMatrix
    : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using VectorType = typename Operator::VectorType;

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
      diagonal_matrix.vmult(dst, src);
    }

    void
    do_update() override
    {
      op.compute_inverse_diagonal(diagonal_matrix.get_vector());
    }

  private:
    const Operator &           op;
    DiagonalMatrix<VectorType> diagonal_matrix;
  };



  template <typename Operator>
  class InverseComponentBlockDiagonalMatrix
    : public PreconditionerBase<typename Operator::value_type>
  {
  public:
    using Number     = typename Operator::value_type;
    using VectorType = typename Operator::VectorType;

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
    using VectorType = typename Operator::VectorType;
    using Number     = typename VectorType::value_type;

    InverseBlockDiagonalMatrix(const Operator &op)
      : op(op)
    {
      AssertThrow(Utilities::MPI::n_mpi_processes(
                    op.get_dof_handler().get_communicator()) == 1,
                  ExcNotImplemented());
    }

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

      for (const auto &cell : op.get_dof_handler().active_cell_iterators())
        {
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
      const auto temp = compute_weights(op.get_dof_handler());

      op.initialize_dof_vector(weights);

      for (unsigned i = 0; i < weights.size(); ++i)
        weights[i] = temp[i];
    }


  private:
    static void
    compute_block_diagonal_matrix(
      const DoFHandler<dim> &                                   dof_handler_0,
      const TrilinosWrappers::SparseMatrix &                    system_matrix_0,
      std::vector<FullMatrix<typename VectorType::value_type>> &blocks)
    {
      const unsigned int dofs_per_cell =
        dof_handler_0.get_fe().n_dofs_per_cell();

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler_0.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          cell->get_dof_indices(local_dof_indices);

          auto &cell_matrix = blocks[cell->active_cell_index()];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) =
                system_matrix_0(local_dof_indices[i], local_dof_indices[j]);

          cell_matrix.gauss_jordan();
        }
    }

    static Vector<Number>
    compute_weights(const DoFHandler<dim> &dof_handler_0)
    {
      const unsigned int dofs_per_cell =
        dof_handler_0.get_fe().n_dofs_per_cell();

      Vector<double>                       weights(dof_handler_0.n_dofs());
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

      for (unsigned int i = 0; i < weights.locally_owned_size(); ++i)
        weights[i] = (weights[i] == 0.0) ? 0.0 : std::sqrt(1.0 / weights[i]);

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
    using VectorType = typename Operator::VectorType;

    AMG(const Operator &op)
      : op(op)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      precondition_amg.vmult(dst, src);
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
    using VectorType = typename Operator::VectorType;

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



  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 20;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;

    unsigned int coarse_grid_smoother_sweeps = 1;
    unsigned int coarse_grid_n_cycles        = 1;
    std::string  coarse_grid_smoother_type   = "ILU";

    unsigned int coarse_grid_maxiter = 1000;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;
  };



  template <int dim, typename LevelMatrixType, typename VectorType>
  class PreconditionerGMG
    : public PreconditionerBase<typename VectorType::value_type>
  {
  public:
    PreconditionerGMG(
      const DoFHandler<dim> &dof_handler,
      const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
        &mg_dof_handlers,
      const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
        &                                                    mg_constraints,
      const MGLevelObject<std::shared_ptr<LevelMatrixType>> &mg_operators)
      : dof_handler(dof_handler)
      , mg_dof_handlers(mg_dof_handlers)
      , mg_constraints(mg_constraints)
      , mg_operators(mg_operators)
      , min_level(mg_dof_handlers.min_level())
      , max_level(mg_dof_handlers.max_level())
      , transfers(min_level, max_level)
      , transfer(transfers, [&](const auto l, auto &vec) {
        this->mg_operators[l]->initialize_dof_vector(vec);
      })
    {
      // setup transfer operators
      for (auto l = min_level; l < max_level; ++l)
        transfers[l + 1].reinit(*mg_dof_handlers[l + 1],
                                *mg_dof_handlers[l],
                                *mg_constraints[l + 1],
                                *mg_constraints[l]);
    }

    void
    do_update() override
    {
      PreconditionerGMGAdditionalData additional_data;

      // wrap level operators
      mg_matrix = std::make_unique<mg::Matrix<VectorType>>(mg_operators);

      // setup smoothers on each level
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        min_level, max_level);

      for (unsigned int level = min_level; level <= max_level; ++level)
        {
          smoother_data[level].preconditioner =
            std::make_shared<SmootherPreconditionerType>();
          mg_operators[level]->compute_inverse_diagonal(
            smoother_data[level].preconditioner->get_vector());
          smoother_data[level].smoothing_range =
            additional_data.smoothing_range;
          smoother_data[level].degree = additional_data.smoothing_degree;
          smoother_data[level].eig_cg_n_iterations =
            additional_data.smoothing_eig_cg_n_iterations;
        }

      mg_smoother.initialize(mg_operators, smoother_data);

      // setup coarse-grid solver
      coarse_grid_solver_control =
        std::make_unique<ReductionControl>(additional_data.coarse_grid_maxiter,
                                           additional_data.coarse_grid_abstol,
                                           additional_data.coarse_grid_reltol,
                                           false,
                                           false);
      coarse_grid_solver =
        std::make_unique<SolverCG<VectorType>>(*coarse_grid_solver_control);

      precondition_amg = std::make_unique<TrilinosWrappers::PreconditionAMG>();

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.smoother_sweeps = additional_data.coarse_grid_smoother_sweeps;
      amg_data.n_cycles        = additional_data.coarse_grid_n_cycles;
      amg_data.smoother_type =
        additional_data.coarse_grid_smoother_type.c_str();
      precondition_amg->initialize(mg_operators[min_level]->get_system_matrix(),
                                   amg_data);
      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    TrilinosWrappers::PreconditionAMG>>(
        *coarse_grid_solver, *mg_operators[min_level], *precondition_amg);

      // create multigrid algorithm (put level operators, smoothers, transfer
      // operators and smoothers together)
      mg = std::make_unique<Multigrid<VectorType>>(*mg_matrix,
                                                   *mg_coarse,
                                                   transfer,
                                                   mg_smoother,
                                                   mg_smoother,
                                                   min_level,
                                                   max_level);

      // convert multigrid algorithm to preconditioner
      preconditioner =
        std::make_unique<PreconditionMG<dim, VectorType, MGTransferType>>(
          dof_handler, *mg, transfer);
    }

    virtual void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      preconditioner->vmult(dst, src);
    }

  private:
    using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;
    using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                               VectorType,
                                               SmootherPreconditionerType>;

    const DoFHandler<dim> &dof_handler;

    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
    const MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
                                                          mg_constraints;
    const MGLevelObject<std::shared_ptr<LevelMatrixType>> mg_operators;

    const unsigned int min_level;
    const unsigned int max_level;

    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
    MGTransferType                                     transfer;

    mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

    mutable MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
      mg_smoother;

    mutable std::unique_ptr<ReductionControl> coarse_grid_solver_control;

    mutable std::unique_ptr<SolverCG<VectorType>> coarse_grid_solver;

    mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

    mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

    mutable std::unique_ptr<Multigrid<VectorType>> mg;

    mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
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
    else if (label == "ILU")
      return std::make_unique<ILU<T>>(op);

    AssertThrow(false,
                ExcMessage("Preconditioner << " + label + " >> not known!"));

    return {};
  }

} // namespace Preconditioners