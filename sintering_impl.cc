// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Sintering of 2 particles

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

//#define WITH_TIMING
//#define WITH_TRACKER

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

namespace dealii
{
  namespace VectorTools
  {
    template <typename VectorType>
    bool
    check_identity(VectorType &vec_0, VectorType &vec_1)
    {
      if (vec_0.get_partitioner()->locally_owned_size() !=
          vec_1.get_partitioner()->locally_owned_size())
        return false;

      for (unsigned int i = 0;
           i < vec_0.get_partitioner()->locally_owned_size();
           ++i)
        if (vec_0.local_element(i) != vec_1.local_element(i))
          return false;

      return true;
    }



    template <typename VectorType>
    void
    split_up_fast(const VectorType &vec, VectorType &vec_0, VectorType &vec_1)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          for (unsigned int j = 0; j < 2; ++j)
            vec_0.local_element(i0++) = vec.local_element(i++);

          for (unsigned int j = 0; j < 2; ++j)
            vec_1.local_element(i1++) = vec.local_element(i++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    split_up(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const VectorType &                                  vec,
             VectorType &                                        vec_0,
             VectorType &                                        vec_1)
    {
      vec.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            // read all dof_values
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());
            cell_all->get_dof_values(vec, local);

            // write Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(1));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c, j)];
                }

              cell.set_dof_values(local_component, vec_0);
            }

            // write Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_1);
            }
          }

      vec.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    split_up_fast(const VectorType &vec,
                  VectorType &      vec_0,
                  VectorType &      vec_1,
                  VectorType &      vec_2)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0, i2 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          vec_0.local_element(i0++) = vec.local_element(i++);
          vec_1.local_element(i1++) = vec.local_element(i++);

          for (unsigned int j = 0; j < 2; ++j)
            vec_2.local_element(i2++) = vec.local_element(i++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    split_up(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const VectorType &                                  vec,
             VectorType &                                        vec_0,
             VectorType &                                        vec_1,
             VectorType &                                        vec_2)
    {
      vec.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            // read all dof_values
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());
            cell_all->get_dof_values(vec, local);

            // write Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c, j)];
                }

              cell.set_dof_values(local_component, vec_0);
            }

            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 1,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_1);
            }

            // write Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_2);
            }
          }

      vec.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    merge_fast(const VectorType &vec_0,
               const VectorType &vec_1,
               VectorType &      vec)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          for (unsigned int j = 0; j < 2; ++j)
            vec.local_element(i++) = vec_0.local_element(i0++);

          for (unsigned int j = 0; j < 2; ++j)
            vec.local_element(i++) = vec_1.local_element(i1++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    merge(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const VectorType &                                  vec_0,
          const VectorType &                                  vec_1,
          VectorType &                                        vec)
    {
      vec_0.update_ghost_values();
      vec_1.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());

            // read Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(1));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_0, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c, j)] =
                    local_component[i];
                }
            }

            // read Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_1, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read all dof_values
            cell_all->set_dof_values(local, vec);
          }

      vec_0.zero_out_ghost_values();
      vec_1.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    merge_fast(const VectorType &vec_0,
               const VectorType &vec_1,
               const VectorType &vec_2,
               VectorType &      vec)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0, i2 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          vec.local_element(i++) = vec_0.local_element(i0++);
          vec.local_element(i++) = vec_1.local_element(i1++);

          for (unsigned int j = 0; j < 2; ++j)
            vec.local_element(i++) = vec_2.local_element(i2++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    merge(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const VectorType &                                  vec_0,
          const VectorType &                                  vec_1,
          const VectorType &                                  vec_2,
          VectorType &                                        vec)
    {
      vec_0.update_ghost_values();
      vec_1.update_ghost_values();
      vec_2.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());

            // read Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_0, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c, j)] =
                    local_component[i];
                }
            }

            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_1, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 1,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_2, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read all dof_values
            cell_all->set_dof_values(local, vec);
          }

      vec_0.zero_out_ghost_values();
      vec_1.zero_out_ghost_values();
      vec_2.zero_out_ghost_values();
    }
  } // namespace VectorTools
} // namespace dealii



class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_, const std::string &section_name)
#ifdef WITH_TIMING
    : scope(timer_, section_name)
#endif
  {
    (void)timer_;
    (void)section_name;
  }

  ~MyScope() = default;

private:
#ifdef WITH_TIMING
  TimerOutput::Scope scope;
#endif
};



namespace Preconditioners
{
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


namespace LinearSolvers
{
  template <typename Number>
  class LinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    virtual ~LinearSolverBase() = default;

    virtual unsigned int
    solve(VectorType &dst, const VectorType &src) = 0;
  };



  template <typename Operator, typename Preconditioner>
  class SolverGMRESWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType = typename Operator::vector_type;

    SolverGMRESWrapper(const Operator &op, Preconditioner &preconditioner)
      : op(op)
      , preconditioner(preconditioner)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {}

    ~SolverGMRESWrapper()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      MyScope scope(timer, "gmres::solve");

      unsigned int            max_iter = 1000;
      ReductionControl        reduction_control(max_iter, 1.e-10, 1.e-2);
      SolverGMRES<VectorType> solver(reduction_control);
      solver.solve(op, dst, src, preconditioner);

      return reduction_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
  };
} // namespace LinearSolvers



namespace NonLinearSolvers
{
  struct NonLinearSolverStatistics
  {
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;
  };



  DeclExceptionMsg(
    ExcNewtonDidNotConverge,
    "Damped Newton iteration did not converge. Maximum number of iterations exceed!");



  struct NewtonSolverData
  {
    NewtonSolverData(const unsigned int max_iter              = 100,
                     const double       abs_tol               = 1.e-20,
                     const double       rel_tol               = 1.e-5,
                     const bool         do_update             = true,
                     const unsigned int threshold_newton_iter = 10,
                     const unsigned int threshold_linear_iter = 20)
      : max_iter(max_iter)
      , abs_tol(abs_tol)
      , rel_tol(rel_tol)
      , do_update(do_update)
      , threshold_newton_iter(threshold_newton_iter)
      , threshold_linear_iter(threshold_linear_iter)
    {}

    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;
    const bool         do_update;
    const unsigned int threshold_newton_iter;
    const unsigned int threshold_linear_iter;
  };



  template <typename VectorType>
  class NewtonSolver
  {
  public:
    NewtonSolver(const NewtonSolverData &solver_data_in = NewtonSolverData())
      : solver_data(solver_data_in)
    {}

    NonLinearSolverStatistics
    solve(VectorType &dst) const
    {
      VectorType vec_residual, increment, tmp;
      reinit_vector(vec_residual);
      reinit_vector(increment);
      reinit_vector(tmp);

      // evaluate residual using the given estimate of the solution
      residual(dst, vec_residual);

      double norm_r   = vec_residual.l2_norm();
      double norm_r_0 = norm_r;

      // Accumulated linear iterations
      NonLinearSolverStatistics statistics;

      unsigned int linear_iterations_last = 0;

      while (norm_r > this->solver_data.abs_tol &&
             norm_r / norm_r_0 > solver_data.rel_tol &&
             statistics.newton_iterations < solver_data.max_iter)
        {
          // reset increment
          increment = 0.0;

          // multiply by -1.0 since the linearized problem is "LinearMatrix *
          // increment = - vec_residual"
          vec_residual *= -1.0;

          // solve linear problem
          bool const threshold_exceeded =
            (statistics.newton_iterations % solver_data.threshold_newton_iter ==
             0) ||
            (linear_iterations_last > solver_data.threshold_linear_iter);

          setup_jacobian(dst, solver_data.do_update && threshold_exceeded);

          linear_iterations_last = solve_with_jacobian(vec_residual, increment);

          statistics.linear_iterations += linear_iterations_last;

          // damped Newton scheme
          const double tau =
            0.1; // another parameter (has to be smaller than 1)
          const unsigned int N_ITER_TMP_MAX =
            100;                   // iteration counts for damping scheme
          double omega      = 1.0; // damping factor
          double norm_r_tmp = 1.0; // norm of residual using temporary solution
          unsigned int n_iter_tmp = 0;

          do
            {
              // calculate temporary solution
              tmp = dst;
              tmp.add(omega, increment);

              // evaluate residual using the temporary solution
              residual(tmp, vec_residual);

              // calculate norm of residual (for temporary solution)
              norm_r_tmp = vec_residual.l2_norm();

              // reduce step length
              omega = omega / 2.0;

              // increment counter
              n_iter_tmp++;
            }
          while (norm_r_tmp >= (1.0 - tau * omega) * norm_r &&
                 n_iter_tmp < N_ITER_TMP_MAX);

          AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                      ExcNewtonDidNotConverge());

          // update solution and residual
          dst    = tmp;
          norm_r = norm_r_tmp;

          // increment iteration counter
          ++statistics.newton_iterations;
        }

      AssertThrow(norm_r <= this->solver_data.abs_tol ||
                    norm_r / norm_r_0 <= solver_data.rel_tol,
                  ExcNewtonDidNotConverge());

      return statistics;
    }


  private:
    const NewtonSolverData solver_data;

  public:
    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
  };
} // namespace NonLinearSolvers



namespace Sintering
{
  template <int dim>
  class InitialValues : public dealii::Function<dim>
  {
  private:
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;

    double r0;
    double interface_width;

    double interface_offset = 0;

    bool is_accumulative;

  public:
    InitialValues(double       x01,
                  double       x02,
                  double       y0,
                  double       r0,
                  double       interface_width,
                  unsigned int n_components,
                  bool         is_accumulative)
      : dealii::Function<dim>(n_components)
      , r0(r0)
      , interface_width(interface_width)
      , is_accumulative(is_accumulative)
    {
      initializeCenters(x01, x02, y0, y0);
    }

    virtual double
    value(const dealii::Point<dim> &p,
          const unsigned int        component = 0) const override
    {
      double ret_val = 0;

      if (component == 0)
        {
          double eta1 = is_in_sphere(p, p1);
          double eta2 = is_in_sphere(p, p2);

          if (is_accumulative)
            {
              ret_val = eta1 + eta2;
            }
          else
            {
              ret_val = std::max(eta1, eta2);
            }
        }
      else if (component == 2)
        {
          ret_val = is_in_sphere(p, p1);
        }
      else if (component == 3)
        {
          ret_val = is_in_sphere(p, p2);
        }
      else
        {
          ret_val = 0;
        }

      return ret_val;
    }

  private:
    void
    initializeCenters(double x01, double x02, double y01, double y02)
    {
      if (dim == 2)
        {
          p1 = dealii::Point<dim>(x01, y01);
          p2 = dealii::Point<dim>(x02, y02);
        }
      else if (dim == 3)
        {
          p1 = dealii::Point<dim>(x01, y01, y01);
          p2 = dealii::Point<dim>(x02, y02, y02);
        }
      else
        {
          throw std::runtime_error("This dim size is not admissible");
        }
    }

    double
    is_in_sphere(const dealii::Point<dim> &point,
                 const dealii::Point<dim> &center) const
    {
      double c = 0;

      double rm  = r0 - interface_offset;
      double rad = center.distance(point);

      if (rad <= rm - interface_width / 2.0)
        {
          c = 1;
        }
      else if (rad < rm + interface_width / 2.0)
        {
          double outvalue = 0.;
          double invalue  = 1.;
          double int_pos = (rad - rm + interface_width / 2.0) / interface_width;

          c = outvalue +
              (invalue - outvalue) * (1.0 + std::cos(int_pos * M_PI)) / 2.0;
          // c = 0.5 - 0.5 * std::sin(M_PI * (rad - rm) / interface_width);
        }

      return c;
    }
  };



  template <int dim, typename VectorizedArrayType>
  class MobilityScalar
  {
  protected:
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;

  public:
    MobilityScalar(const double Mvol,
                   const double Mvap,
                   const double Msurf,
                   const double Mgb)
      : Mvol(Mvol)
      , Mvap(Mvap)
      , Msurf(Msurf)
      , Mgb(Mgb)
    {}

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M(const VectorizedArrayType &                               c,
      const std::array<VectorizedArrayType, n> &                etas,
      const Tensor<1, dim, VectorizedArrayType> &               c_grad,
      const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad) const
    {
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType etaijSum = 0.0;
      for (const auto &etai : etas)
        {
          for (const auto &etaj : etas)
            {
              if (&etai != &etaj)
                {
                  etaijSum += etai * etaj;
                }
            }
        }

      VectorizedArrayType phi =
        cl * cl * cl * (10.0 - 15.0 * cl + 6.0 * cl * cl);
      std::for_each(phi.begin(), phi.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType M = Mvol * phi + Mvap * (1.0 - phi) +
                              Msurf * cl * (1.0 - cl) + Mgb * etaijSum;

      return M;
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_dc(
      const VectorizedArrayType &                               c,
      const std::array<VectorizedArrayType, n> &                etas,
      const Tensor<1, dim, VectorizedArrayType> &               c_grad,
      const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad) const
    {
      (void)etas;
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType dphidc = 30.0 * cl * cl * (1.0 - 2.0 * cl + cl * cl);
      VectorizedArrayType dMdc =
        Mvol * dphidc - Mvap * dphidc + Msurf * (1.0 - 2.0 * cl);

      return dMdc;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      (void)c;
      (void)c_grad;
      (void)mu_grad;

      return Tensor<2, dim, VectorizedArrayType>();
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_detai(
      const VectorizedArrayType &                               c,
      const std::array<VectorizedArrayType, n> &                etas,
      const Tensor<1, dim, VectorizedArrayType> &               c_grad,
      const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad,
      unsigned int                                              index_i) const
    {
      (void)c;
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType etajSum = 0;
      for (unsigned int j = 0; j < etas.size(); j++)
        {
          if (j != index_i)
            {
              etajSum += etas[j];
            }
        }

      auto MetajSum = 2.0 * Mgb * etajSum;

      return MetajSum;
    }
  };



  template <int dim, typename VectorizedArrayType>
  class MobilityTensorial
  {
  protected:
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;

  public:
    MobilityTensorial(const double Mvol,
                      const double Mvap,
                      const double Msurf,
                      const double Mgb)
      : Mvol(Mvol)
      , Mvap(Mvap)
      , Msurf(Msurf)
      , Mgb(Mgb)
    {}

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M(const VectorizedArrayType &                               c,
                            const std::array<VectorizedArrayType, n> &                etas,
                            const Tensor<1, dim, VectorizedArrayType> &               c_grad,
                            const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad) const
    {
      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType phi =
        cl * cl * cl * (10.0 - 15.0 * cl + 6.0 * cl * cl);
      std::for_each(phi.begin(), phi.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M =
        unitMatrix(Mvol * phi + Mvap * (1.0 - phi));

      // Surface anisotropic part
      VectorizedArrayType fsurf = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      M += projectorMatrix(nc, fsurf);

      // GB diffusion part
      for (unsigned int i = 0; i < etas.size(); i++)
        {
          for (unsigned int j = 0; j < etas.size(); j++)
            {
              if (i != j)
                {
                  VectorizedArrayType fgb = Mgb * etas[i] * etas[j];
                  Tensor<1, dim, VectorizedArrayType> etaGradDiff =
                    etas_grad[i] - etas_grad[j];
                  Tensor<1, dim, VectorizedArrayType> neta =
                    unitVector(etaGradDiff);
                  M += projectorMatrix(neta, fgb);
                }
            }
        }

      return M;
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dc(
                            const VectorizedArrayType &                               c,
                            const std::array<VectorizedArrayType, n> &                etas,
                            const Tensor<1, dim, VectorizedArrayType> &               c_grad,
                            const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad) const
    {
      (void)etas;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType dphidc = 30.0 * cl * cl * (1.0 - 2.0 * cl + cl * cl);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> dMdc =
        unitMatrix((Mvol - Mvap) * dphidc);

      // Surface part
      VectorizedArrayType fsurf  = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      VectorizedArrayType dfsurf = Msurf * 2. * cl * (1. - cl) * (1. - 2. * cl);
      for (unsigned int i = 0; i < fsurf.size(); i++)
        {
          if (fsurf[i] < 1e-6)
            {
              dfsurf[i] = 0.;
            }
        }
      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      dMdc += projectorMatrix(nc, dfsurf);

      return dMdc;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType fsurf = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      VectorizedArrayType nrm   = c_grad.norm();

      for (unsigned int i = 0; i < nrm.size(); i++)
        {
          if (nrm[i] < 1e-4 || fsurf[i] < 1e-6)
            {
              fsurf[i] = 0.;
            }
          if (nrm[i] < 1e-10)
            {
              nrm[i] = 1.;
            }
        }

      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      Tensor<2, dim, VectorizedArrayType> M  = projectorMatrix(nc, 1. / nrm);

      Tensor<2, dim, VectorizedArrayType> T =
        unitMatrix(mu_grad * nc) + outer_product(nc, mu_grad);
      T *= -fsurf;

      return T * M;
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_detai(
                            const VectorizedArrayType &                               c,
                            const std::array<VectorizedArrayType, n> &                etas,
                            const Tensor<1, dim, VectorizedArrayType> &               c_grad,
                            const std::array<Tensor<1, dim, VectorizedArrayType>, n> &etas_grad,
                            unsigned int                                              index_i) const
    {
      (void)c;
      (void)c_grad;

      dealii::Tensor<2, dim, VectorizedArrayType> M;

      for (unsigned int j = 0; j < etas.size(); j++)
        {
          if (j != index_i)
            {
              VectorizedArrayType                 fgb = 2. * Mgb * etas[j];
              Tensor<1, dim, VectorizedArrayType> etaGradDiff =
                etas_grad[index_i] - etas_grad[j];
              Tensor<1, dim, VectorizedArrayType> neta =
                unitVector(etaGradDiff);
              M += projectorMatrix(neta, fgb);
            }
        }

      return M;
    }

  private:
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          unitMatrix(const VectorizedArrayType &fac = 1.) const
    {
      Tensor<2, dim, VectorizedArrayType> I;

      for (unsigned int d = 0; d < dim; d++)
        {
          I[d][d] = fac;
        }

      return I;
    }

    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
    unitVector(const Tensor<1, dim, VectorizedArrayType> &vec) const
    {
      VectorizedArrayType nrm = vec.norm();
      VectorizedArrayType filter;

      Tensor<1, dim, VectorizedArrayType> n = vec;

      for (unsigned int i = 0; i < nrm.size(); i++)
        {
          if (nrm[i] > 1e-4)
            {
              filter[i] = 1.;
            }
          else
            {
              nrm[i] = 1.;
            }
        }

      n /= nrm;
      n *= filter;

      return n;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
    projectorMatrix(const Tensor<1, dim, VectorizedArrayType> vec,
                    const VectorizedArrayType &               fac = 1.) const
    {
      auto tensor = unitMatrix() - dealii::outer_product(vec, vec);
      tensor *= fac;

      return tensor;
    }
  };

  template <unsigned int n, std::size_t p>
  class PowerHelper
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, n> &etas)
    {
      T initial = 0.0;

      return std::accumulate(etas.begin(),
                             etas.end(),
                             initial,
                             [](auto a, auto b) {
                               return std::move(a) + std::pow(b, n);
                             });
    }
  };

  template <>
  class PowerHelper<2, 2>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }
  };

  template <>
  class PowerHelper<2, 3>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }
  };

  template <typename VectorizedArrayType>
  class FreeEnergy
  {
  private:
    double A;
    double B;

  public:
    FreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &               c,
      const std::array<VectorizedArrayType, n> &etas) const
    {
      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
             B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                  (-4.0 * c + 8.0) * etaPower3Sum +
                  3.0 * (etaPower2Sum * etaPower2Sum));
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &               c,
          const std::array<VectorizedArrayType, n> &etas) const
    {
      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return A * (c * c) * (2.0 * c - 2.0) +
             2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
             B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &               c,
             const std::array<VectorizedArrayType, n> &etas,
             unsigned int                              index_i) const
    {
      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      const auto &etai = etas[index_i];

      return B * (3.0 * (etai * etai) * (4.0 * c - 8.0) +
                  2.0 * etai * (-6.0 * c + 6.0) + 12.0 * etai * (etaPower2Sum));
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dc2(const VectorizedArrayType &               c,
            const std::array<VectorizedArrayType, n> &etas) const
    {
      (void)etas;

      return 2.0 * A * (c * c) + 4.0 * A * c * (2.0 * c - 2.0) +
             2.0 * A * ((-c + 1.0) * (-c + 1.0)) + 2.0 * B;
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dcdetai(const VectorizedArrayType &               c,
                const std::array<VectorizedArrayType, n> &etas,
                unsigned int                              index_i) const
    {
      (void)c;

      const auto &etai = etas[index_i];

      return B * (12.0 * (etai * etai) - 12.0 * etai);
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &               c,
               const std::array<VectorizedArrayType, n> &etas,
               unsigned int                              index_i) const
    {
      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      const auto &etai = etas[index_i];

      return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                  24.0 * (etai * etai) + 12.0 * etaPower2Sum);
    }

    template <std::size_t n>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detaidetaj(const VectorizedArrayType &               c,
                   const std::array<VectorizedArrayType, n> &etas,
                   unsigned int                              index_i,
                   unsigned int                              index_j) const
    {
      (void)c;

      const auto &etai = etas[index_i];
      const auto &etaj = etas[index_j];

      return 24.0 * B * etai * etaj;
    }
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class OperatorBase : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    static const int dimension = dim;

    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const unsigned int                                    dof_index,
      const std::string                                     label = "")
      : matrix_free(matrix_free)
      , constraints(*constraints[dof_index])
      , dof_index(dof_index)
      , label(label)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {}

    virtual ~OperatorBase()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    virtual void
    clear()
    {
      this->system_matrix.clear();
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return matrix_free.get_dof_handler(dof_index);
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst, dof_index);
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler(dof_index).n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult");


      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          matrix_free.cell_loop(
            &OperatorBase::do_vmult_range, this, dst, src, true);
        }
      else
        {
          system_matrix.vmult(dst, src);
        }
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      AssertThrow(false, ExcNotImplemented());

      this->vmult(dst, src);
    }

    void
    compute_inverse_diagonal(VectorType &diagonal) const
    {
      MyScope scope(this->timer, label + "::diagonal");

      matrix_free.initialize_dof_vector(diagonal, dof_index);
      MatrixFreeTools::compute_diagonal(
        matrix_free, diagonal, &OperatorBase::do_vmult_cell, this, dof_index);
      for (auto &i : diagonal)
        i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer, label + "::matrix::sp");

          system_matrix.clear();

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(),
            dof_handler.get_triangulation().get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
          dsp.compress();

          system_matrix.reinit(dsp);
        }

      {
        MyScope scope(this->timer, label + "::matrix::compute");

        if (system_matrix_is_empty == false)
          system_matrix = 0.0; // clear existing content

        MatrixFreeTools::compute_matrix(matrix_free,
                                        constraints,
                                        system_matrix,
                                        &OperatorBase::do_vmult_cell,
                                        this,
                                        dof_index);
      }

      return system_matrix;
    }

  protected:
    virtual void
    do_vmult_kernel(FECellIntegrator &phi) const = 0;

    void
    do_vmult_cell(FECellIntegrator &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                   EvaluationFlags::EvaluationFlags::gradients);

      do_vmult_kernel(phi);

      phi.integrate(EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);
    }

    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator phi(matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          do_vmult_kernel(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

  protected:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const AffineConstraints<Number> &                   constraints;

    const unsigned int dof_index;
    const std::string  label;

    mutable TrilinosWrappers::SparseMatrix system_matrix;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
  };



  template <typename Number, typename VectorizedArrayType>
  class ConstantsTracker
  {
  public:
    ConstantsTracker()
      : n_th(1)
      , max_level(2)
    {}

    void
    initialize(const unsigned int n_filled_lanes)
    {
      this->n_filled_lanes = n_filled_lanes;
      this->temp_min.clear();
      this->temp_max.clear();
    }

    void
    emplace_back(const unsigned int level, const Number &value)
    {
      if (level > max_level)
        return;

      temp_min.emplace_back(value);
      temp_max.emplace_back(value);
    }

    void
    emplace_back(const unsigned int level, const VectorizedArrayType &value)
    {
      if (level > max_level)
        return;

      const auto [min_value, max_value] = get_min_max(value);
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
    }

    template <int dim>
    void
    emplace_back(const unsigned int                         level,
                 const Tensor<1, dim, VectorizedArrayType> &value)
    {
      if (level > max_level)
        return;

#if false
      for (unsigned int d = 0; d < dim; ++d)
          {
            const auto [min_value, max_value] = get_min_max(value[d]);
            temp_min.emplace_back(min_value);
            temp_max.emplace_back(max_value);
          }
#else
      const auto [min_value, max_value] = get_min_max(value.norm());
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
#endif
    }

    template <int dim>
    void
    emplace_back(const unsigned int                         level,
                 const Tensor<2, dim, VectorizedArrayType> &value)
    {
      if (level > max_level)
        return;

#if false
      for (unsigned int d0 = 0; d0 < dim; ++d0)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
            {
              const auto [min_value, max_value] = get_min_max(value[d0][d1]);
              temp_min.emplace_back(min_value);
              temp_max.emplace_back(max_value);
            }
#else
      const auto [min_value, max_value] = get_min_max(value.norm());
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
#endif
    }

    void
    finalize_point()
    {
      if (temp_min_0.size() == 0)
        temp_min_0 = temp_min;
      else
        {
          for (unsigned int i = 0; i < temp_min_0.size(); ++i)
            temp_min_0[i] = std::min(temp_min_0[i], temp_min[i]);
        }

      if (temp_max_0.size() == 0)
        temp_max_0 = temp_max;
      else
        {
          for (unsigned int i = 0; i < temp_max_0.size(); ++i)
            temp_max_0[i] = std::max(temp_max_0[i], temp_max[i]);
        }

      temp_min.clear();
      temp_max.clear();
    }

    void
    finalize()
    {
      std::vector<Number> global_min(temp_min_0.size());
      Utilities::MPI::min(temp_min_0, MPI_COMM_WORLD, global_min);
      all_values_min.emplace_back(global_min);

      std::vector<Number> global_max(temp_max_0.size());
      Utilities::MPI::max(temp_max_0, MPI_COMM_WORLD, global_max);
      all_values_max.emplace_back(global_max);

      this->temp_min_0.clear();
      this->temp_max_0.clear();
    }

    void
    print()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          const auto internal_print = [this](const auto &all_values,
                                             const auto &label) {
            std::ofstream pcout;
            pcout.open(label);

            unsigned int i = 0;

            for (; i < all_values.size() - 1; i += n_th)
              {
                for (unsigned int j = 0; j < all_values[i].size(); ++j)
                  pcout << all_values[i][j] << " ";

                pcout << std::endl;
              }

            if (n_th != 0)
              if ((i + 2) != (all_values.size() + n_th)) // print last entry
                {
                  i = all_values.size() - 2;

                  for (unsigned int j = 0; j < all_values[i].size(); ++j)
                    pcout << all_values[i][j] << " ";

                  pcout << std::endl;
                }

            pcout.close();
          };

          internal_print(all_values_min, "constants_min.txt");
          internal_print(all_values_max, "constants_max.txt");
        }
    }

  private:
    std::pair<Number, Number>
    get_min_max(const VectorizedArrayType &value) const
    {
      Number min_val = 0;
      Number max_val = 0;

      for (unsigned int i = 0; i < n_filled_lanes; ++i)
        {
          const auto val = value[i];

          if (i == 0)
            {
              min_val = val;
              max_val = val;
            }
          else
            {
              min_val = std::min(val, min_val);
              max_val = std::max(val, max_val);
            }
        }

      return {min_val, max_val};
    }

    unsigned int        n_filled_lanes;
    std::vector<Number> temp_min;
    std::vector<Number> temp_max;

    std::vector<Number> temp_min_0;
    std::vector<Number> temp_max_0;

    std::vector<std::vector<Number>> all_values_min;
    std::vector<std::vector<Number>> all_values_max;

    const unsigned int n_th;
    const unsigned int max_level;
  };



  template <int dim, typename VectorizedArrayType>
  struct SinteringOperatorData
  {
    using Number = typename VectorizedArrayType::value_type;

    SinteringOperatorData(const Number A,
                          const Number B,
                          const Number Mvol,
                          const Number Mvap,
                          const Number Msurf,
                          const Number Mgb,
                          const Number L,
                          const Number kappa_c,
                          const Number kappa_p)
      : free_energy(A, B)
      , mobility(Mvol, Mvap, Msurf, Mgb)
      , L(L)
      , kappa_c(kappa_c)
      , kappa_p(kappa_p)
    {}

    const FreeEnergy<VectorizedArrayType> free_energy;

    // Choose MobilityScalar or MobilityTensorial here:
    const MobilityScalar<dim, VectorizedArrayType> mobility;
    // const MobilityTensorial<dim, VectorizedArrayType> mobility;

    const Number L;
    const Number kappa_c;
    const Number kappa_p;
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class SinteringOperator
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    SinteringOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const std::vector<const AffineConstraints<Number> *> & constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const bool                                             matrix_based)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          0,
          "sintering_op")
      , data(data)
      , phi_lin(this->matrix_free, this->dof_index)
      , matrix_based(matrix_based)
    {}

    ~SinteringOperator()
    {
#ifdef WITH_TRACKER
      tracker.print();
#endif
    }

    void
    evaluate_nonlinear_residual(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, "sintering_op::nonlinear_residual");

      this->matrix_free.cell_loop(
        &SinteringOperator::do_evaluate_nonlinear_residual,
        this,
        dst,
        src,
        true);
    }

    void
    set_previous_solution(const VectorType &src) const
    {
      this->old_solution = src;
      this->old_solution.update_ghost_values();
    }

    const VectorType &
    get_previous_solution() const
    {
      this->old_solution.zero_out_ghost_values();
      return this->old_solution;
    }

    void
    evaluate_newton_step(const VectorType &newton_step)
    {
      MyScope scope(this->timer, "sintering_op::newton_step");

      const unsigned n_cells = this->matrix_free.n_cell_batches();
      const unsigned n_quadrature_points =
        this->matrix_free.get_quadrature().size();

      nonlinear_values.reinit(n_cells, n_quadrature_points);
      nonlinear_gradients.reinit(n_cells, n_quadrature_points);

      int dummy = 0;

      this->matrix_free.cell_loop(&SinteringOperator::do_evaluate_newton_step,
                                  this,
                                  dummy,
                                  newton_step);

#ifdef WITH_TRACKER
      tracker.finalize();
#endif

      this->newton_step = newton_step;
      this->newton_step.update_ghost_values();

      if (matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    void
    set_timestep(double dt_new)
    {
      this->dt = dt_new;
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &
    get_data() const
    {
      return data;
    }

    const double &
    get_dt() const
    {
      return this->dt;
    }

    const Table<2, dealii::Tensor<1, n_components, VectorizedArrayType>> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }


    const Table<2,
                dealii::Tensor<1,
                               n_components,
                               dealii::Tensor<1, dim, VectorizedArrayType>>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
    }

    void
    add_data_vectors(DataOut<dim> &data_out, const VectorType &vec) const
    {
      constexpr unsigned int            n_entries = 17;
      std::array<VectorType, n_entries> data_vectors;

      for (auto &data_vector : data_vectors)
        this->matrix_free.initialize_dof_vector(data_vector, 3);

      FECellIntegrator fe_eval_all(this->matrix_free);
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> fe_eval(
        this->matrix_free, 3 /*scalar dof index*/);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, 1, Number, VectorizedArrayType>
          inverse_mass_matrix(fe_eval);

      AlignedVector<VectorizedArrayType> buffer(fe_eval.n_q_points * n_entries);

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / dt;

      vec.update_ghost_values();

      for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
           ++cell)
        {
          fe_eval_all.reinit(cell);
          fe_eval.reinit(cell);

          fe_eval_all.reinit(cell);
          fe_eval_all.read_dof_values_plain(vec);
          fe_eval_all.evaluate(EvaluationFlags::values |
                               EvaluationFlags::gradients);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              const auto c    = fe_eval_all.get_value(q)[0];
              const auto eta1 = fe_eval_all.get_value(q)[2];
              const auto eta2 = fe_eval_all.get_value(q)[3];

              const auto c_grad    = fe_eval_all.get_gradient(q)[0];
              const auto mu_grad   = fe_eval_all.get_gradient(q)[1];
              const auto eta1_grad = fe_eval_all.get_gradient(q)[2];
              const auto eta2_grad = fe_eval_all.get_gradient(q)[3];

              const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
              const std::array<Tensor<1, dim, VectorizedArrayType>, 2>
                etas_grad{{eta1_grad, eta2_grad}};

              // clang-format off
              const std::array<VectorizedArrayType, n_entries> temp{{
                 VectorizedArrayType(dt_inv),                                         // 00
                 free_energy.d2f_dc2(c, etas),                                        // 01
                 free_energy.d2f_dcdetai(c, etas, 0),                                 // 02
                 free_energy.d2f_dcdetai(c, etas, 1),                                 // 03
                 L * free_energy.d2f_dcdetai(c, etas, 0),                             // 04
                 L * free_energy.d2f_detai2(c, etas, 0),                              // 05
                 L * free_energy.d2f_detaidetaj(c, etas, 0, 1),                       // 06
                 L * free_energy.d2f_dcdetai(c, etas, 1),                             // 07
                 L * free_energy.d2f_detaidetaj(c, etas, 1, 0),                       // 08
                 L * free_energy.d2f_detai2(c, etas, 1),                              // 09
                 mobility.M(c, etas, c_grad, etas_grad),                              // 10
                 (mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad).norm(),       // 11
                 (mobility.dM_dgrad_c(c, c_grad, mu_grad)).norm(),                    // 12
                 (mobility.dM_detai(c, etas, c_grad, etas_grad, 0) * mu_grad).norm(), // 13
                 (mobility.dM_detai(c, etas, c_grad, etas_grad, 1) * mu_grad).norm(), // 14
                 VectorizedArrayType(kappa_c),                                        // 15
                 VectorizedArrayType(L * kappa_p)                                     // 16
                 }};
              // clang-format on

              for (unsigned int c = 0; c < n_entries; ++c)
                buffer[c * fe_eval.n_q_points + q] = temp[c];
            }

          for (unsigned int c = 0; c < n_entries; ++c)
            {
              inverse_mass_matrix.transform_from_q_points_to_basis(
                1,
                buffer.data() + c * fe_eval.n_q_points,
                fe_eval.begin_dof_values());

              fe_eval.set_dof_values(data_vectors[c]);
            }
        }

      vec.zero_out_ghost_values();

      for (unsigned int c = 0; c < n_entries; ++c)
        {
          std::ostringstream ss;
          ss << "aux_" << std::setw(2) << std::setfill('0') << c;

          data_out.add_data_vector(this->matrix_free.get_dof_handler(3),
                                   data_vectors[c],
                                   ss.str());
        }
    }

  private:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / dt;

#if true
      phi_lin.reinit(cell);
      phi_lin.read_dof_values_plain(this->newton_step);
      phi_lin.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
#endif

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
#if true
          const auto c    = phi_lin.get_value(q)[0];
          const auto eta1 = phi_lin.get_value(q)[2];
          const auto eta2 = phi_lin.get_value(q)[3];

          const auto c_grad    = phi_lin.get_gradient(q)[0];
          const auto mu_grad   = phi_lin.get_gradient(q)[1];
          const auto eta1_grad = phi_lin.get_gradient(q)[2];
          const auto eta2_grad = phi_lin.get_gradient(q)[3];
#else
          const auto &c    = nonlinear_values(cell, q)[0];
          const auto &eta1 = nonlinear_values(cell, q)[2];
          const auto &eta2 = nonlinear_values(cell, q)[3];

          const auto &c_grad    = nonlinear_gradients(cell, q)[0];
          const auto &mu_grad   = nonlinear_gradients(cell, q)[1];
          const auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          const auto &eta2_grad = nonlinear_gradients(cell, q)[3];
#endif

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          Tensor<1, n_components, VectorizedArrayType> value_result;

          Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          value_result[0] = phi.get_value(q)[0] * dt_inv;
          value_result[1] =
            -phi.get_value(q)[1] +
            free_energy.d2f_dc2(c, etas) * phi.get_value(q)[0] +
            free_energy.d2f_dcdetai(c, etas, 0) * phi.get_value(q)[2] +
            free_energy.d2f_dcdetai(c, etas, 1) * phi.get_value(q)[3];
          value_result[2] =
            phi.get_value(q)[2] * dt_inv +
            L * free_energy.d2f_dcdetai(c, etas, 0) * phi.get_value(q)[0] +
            L * free_energy.d2f_detai2(c, etas, 0) * phi.get_value(q)[2] +
            L * free_energy.d2f_detaidetaj(c, etas, 0, 1) * phi.get_value(q)[3];
          value_result[3] =
            phi.get_value(q)[3] * dt_inv +
            L * free_energy.d2f_dcdetai(c, etas, 1) * phi.get_value(q)[0] +
            L * free_energy.d2f_detaidetaj(c, etas, 1, 0) *
              phi.get_value(q)[2] +
            L * free_energy.d2f_detai2(c, etas, 1) * phi.get_value(q)[3];

          gradient_result[0] =
            mobility.M(c, etas, c_grad, etas_grad) * phi.get_gradient(q)[1] +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad *
              phi.get_value(q)[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * phi.get_gradient(q)[0] +
            mobility.dM_detai(c, etas, c_grad, etas_grad, 0) * mu_grad *
              phi.get_value(q)[2] +
            mobility.dM_detai(c, etas, c_grad, etas_grad, 1) * mu_grad *
              phi.get_value(q)[3];
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
          gradient_result[2] = L * kappa_p * phi.get_gradient(q)[2];
          gradient_result[3] = L * kappa_p * phi.get_gradient(q)[3];

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator phi_old(matrix_free);
      FECellIntegrator phi(matrix_free);

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_old.reinit(cell);
          phi.reinit(cell);

          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          // get values from old solution
          phi_old.read_dof_values_plain(old_solution);
          phi_old.evaluate(EvaluationFlags::EvaluationFlags::values);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto val     = phi.get_value(q);
              const auto val_old = phi_old.get_value(q);
              const auto grad    = phi.get_gradient(q);

              auto &c    = val[0];
              auto &mu   = val[1];
              auto &eta1 = val[2];
              auto &eta2 = val[3];

              auto &c_old    = val_old[0];
              auto &eta1_old = val_old[2];
              auto &eta2_old = val_old[3];

              auto &c_grad    = grad[0];
              auto &eta1_grad = grad[2];
              auto &eta2_grad = grad[3];

              const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
              const std::array<Tensor<1, dim, VectorizedArrayType>, 2>
                etas_grad{{eta1_grad, eta2_grad}};

              Tensor<1, n_components, VectorizedArrayType> value_result;

              value_result[0] = (c - c_old) / dt;
              value_result[1] = -mu + free_energy.df_dc(c, etas);
              value_result[2] =
                (eta1 - eta1_old) / dt + L * free_energy.df_detai(c, etas, 0);
              value_result[3] =
                (eta2 - eta2_old) / dt + L * free_energy.df_detai(c, etas, 1);

              Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
                gradient_result;

              gradient_result[0] =
                mobility.M(c, etas, c_grad, etas_grad) * grad[1];
              gradient_result[1] = kappa_c * grad[0];
              gradient_result[2] = L * kappa_p * grad[2];
              gradient_result[3] = L * kappa_p * grad[3];

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    void
    do_evaluate_newton_step(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      int &,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &range)
    {
#ifdef WITH_TRACKER
      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / dt;
#endif

      FECellIntegrator phi(matrix_free);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values_plain(src);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

#ifdef WITH_TRACKER
          tracker.initialize(matrix_free.n_active_entries_per_cell_batch(cell));
#endif

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              nonlinear_values(cell, q)    = phi.get_value(q);
              nonlinear_gradients(cell, q) = phi.get_gradient(q);

#ifdef WITH_TRACKER
              const auto &c    = nonlinear_values(cell, q)[0];
              const auto &eta1 = nonlinear_values(cell, q)[2];
              const auto &eta2 = nonlinear_values(cell, q)[3];

              const auto &c_grad    = nonlinear_gradients(cell, q)[0];
              const auto &mu_grad   = nonlinear_gradients(cell, q)[1];
              const auto &eta1_grad = nonlinear_gradients(cell, q)[2];
              const auto &eta2_grad = nonlinear_gradients(cell, q)[3];

              const std::array<VectorizedArrayType, 2> etas{{ eta1, eta2 }};
              const std::array<Tensor<1, dim, VectorizedArrayType>, 2>
                etas_grad{
                  { eta1_grad,
                    eta2_grad }};

              // clang-format off
              tracker.emplace_back(0, dt_inv);
              tracker.emplace_back(1, free_energy.d2f_dc2(c, etas));
              tracker.emplace_back(2, free_energy.d2f_dcdetai(c, etas, 0));
              tracker.emplace_back(2, free_energy.d2f_dcdetai(c, etas, 1));
              tracker.emplace_back(2, L * free_energy.d2f_dcdetai(c, etas, 0));
              tracker.emplace_back(2, L * free_energy.d2f_detai2(c, etas, 0));
              tracker.emplace_back(2, L * free_energy.d2f_detaidetaj(c, etas, 0, 1));
              tracker.emplace_back(2, L * free_energy.d2f_dcdetai(c, etas, 1));
              tracker.emplace_back(2, L * free_energy.d2f_detaidetaj(c, etas, 1, 0));
              tracker.emplace_back(2, L * free_energy.d2f_detai2(c, etas, 1));
              tracker.emplace_back(0, mobility.M(c, etas, c_grad, etas_grad));
              tracker.emplace_back(1, mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad);
              tracker.emplace_back(1, mobility.dM_dgrad_c(c, c_grad, mu_grad));
              tracker.emplace_back(2, mobility.dM_detai(c, etas, c_grad, etas_grad, 0) * mu_grad);
              tracker.emplace_back(2, mobility.dM_detai(c, etas, c_grad, etas_grad, 1) * mu_grad); 
              tracker.emplace_back(0, kappa_c);
              tracker.emplace_back(2, L * kappa_p);
              // clang-format on

              tracker.finalize_point();
#endif
            }
        }
    }

    SinteringOperatorData<dim, VectorizedArrayType> data;

    double dt;

    mutable VectorType old_solution, newton_step;

    mutable FECellIntegrator phi_lin;

    Table<2, dealii::Tensor<1, n_components, VectorizedArrayType>>
      nonlinear_values;
    Table<2,
          dealii::Tensor<1,
                         n_components,
                         dealii::Tensor<1, dim, VectorizedArrayType>>>
      nonlinear_gradients;

    ConstantsTracker<Number, VectorizedArrayType> tracker;

    const bool matrix_based;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliard
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorCahnHilliard(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          1,
          "cahn_hilliard_op")
      , op(op)
    {}

  protected:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy         = this->op.get_data().free_energy;
      const auto &mobility            = this->op.get_data().mobility;
      const auto &kappa_c             = this->op.get_data().kappa_c;
      const auto &dt                  = this->op.get_dt();
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          auto &c    = nonlinear_values(cell, q)[0];
          auto &eta1 = nonlinear_values(cell, q)[2];
          auto &eta2 = nonlinear_values(cell, q)[3];

          auto &c_grad    = nonlinear_gradients(cell, q)[0];
          auto &mu_grad   = nonlinear_gradients(cell, q)[1];
          auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          auto &eta2_grad = nonlinear_gradients(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          Tensor<1, n_components, VectorizedArrayType> value_result;
          Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

#if true
          // CH with all terms
          value_result[0] = phi.get_value(q)[0] / dt;
          value_result[1] = -phi.get_value(q)[1] +
                            free_energy.d2f_dc2(c, etas) * phi.get_value(q)[0];

          gradient_result[0] =
            mobility.M(c, etas, c_grad, etas_grad) * phi.get_gradient(q)[1] +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad *
              phi.get_value(q)[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * phi.get_gradient(q)[0];
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
#else
          // CH with the terms as considered in BlockPreconditioner3CHData
          value_result[0] = phi.get_value(q)[0] / dt;
          value_result[1] = -phi.get_value(q)[1];

          gradient_result[0] =
            mobility.M(c, etas, c_grad, etas_grad) * phi.get_gradient(q)[1];
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
#endif

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliardA
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorCahnHilliardA(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

  protected:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &dt                  = this->op.get_dt();
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          auto &c    = nonlinear_values(cell, q)[0];
          auto &eta1 = nonlinear_values(cell, q)[2];
          auto &eta2 = nonlinear_values(cell, q)[3];

          auto &c_grad    = nonlinear_gradients(cell, q)[0];
          auto &mu_grad   = nonlinear_gradients(cell, q)[1];
          auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          auto &eta2_grad = nonlinear_gradients(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(value / dt, q);
          phi.submit_gradient(mobility.dM_dc(c, etas, c_grad, etas_grad) *
                                  mu_grad * value +
                                mobility.dM_dgrad_c(c, c_grad, mu_grad) *
                                  gradient,
                              q);
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliardB
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorCahnHilliardB(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

  protected:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          auto &c    = nonlinear_values(cell, q)[0];
          auto &eta1 = nonlinear_values(cell, q)[2];
          auto &eta2 = nonlinear_values(cell, q)[3];

          auto &c_grad    = nonlinear_gradients(cell, q)[0];
          auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          auto &eta2_grad = nonlinear_gradients(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(value * 0.0, q); // TODO
          phi.submit_gradient(mobility.M(c, etas, c_grad, etas_grad) * gradient,
                              q);
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliardC
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorCahnHilliardC(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

  protected:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy      = this->op.get_data().free_energy;
      const auto &kappa_c          = this->op.get_data().kappa_c;
      const auto &nonlinear_values = this->op.get_nonlinear_values();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          auto &c    = nonlinear_values(cell, q)[0];
          auto &eta1 = nonlinear_values(cell, q)[2];
          auto &eta2 = nonlinear_values(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(free_energy.d2f_dc2(c, etas) * value, q);
          phi.submit_gradient(kappa_c * gradient, q);
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliardD
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorCahnHilliardD(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

  protected:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(-value, q);
          phi.submit_gradient(gradient * 0.0, q); // TODO
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorAllenCahn
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    OperatorAllenCahn(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          2,
          "allen_cahn_op")
      , op(op)
    {}

  private:
    void
    do_vmult_kernel(FECellIntegrator &phi) const final
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy      = this->op.get_data().free_energy;
      const auto &L                = this->op.get_data().L;
      const auto &kappa_p          = this->op.get_data().kappa_p;
      const auto &dt               = this->op.get_dt();
      const auto &nonlinear_values = this->op.get_nonlinear_values();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          auto &c    = nonlinear_values(cell, q)[0];
          auto &eta1 = nonlinear_values(cell, q)[2];
          auto &eta2 = nonlinear_values(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};

          Tensor<1, n_components, VectorizedArrayType> value_result;

          value_result[0] =
            phi.get_value(q)[0] / dt +
            L * free_energy.d2f_detai2(c, etas, 0) * phi.get_value(q)[0] +
            L * free_energy.d2f_detaidetaj(c, etas, 0, 1) * phi.get_value(q)[1];
          value_result[1] =
            phi.get_value(q)[1] / dt +
            L * free_energy.d2f_detaidetaj(c, etas, 1, 0) *
              phi.get_value(q)[0] +
            L * free_energy.d2f_detai2(c, etas, 1) * phi.get_value(q)[1];

          Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          gradient_result[0] = L * kappa_p * phi.get_gradient(q)[0];
          gradient_result[1] = L * kappa_p * phi.get_gradient(q)[1];

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorAllenCahnHelmholtz
    : public OperatorBase<dim, 1, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

    using VectorType =
      typename OperatorBase<dim, 1, Number, VectorizedArrayType>::VectorType;

    OperatorAllenCahnHelmholtz(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, 1, Number, VectorizedArrayType>(matrix_free,
                                                          constraints,
                                                          3,
                                                          "helmholtz_op")
      , op(op)
    {}

    double
    get_dt() const
    {
      return op.get_dt();
    }

    void
    get_diagonals(VectorType &vec_mass, VectorType &vec_laplace) const
    {
      {
        MyScope scope(this->timer, "helmholtz_op::get_diagonal::mass");
        this->initialize_dof_vector(vec_mass);
        MatrixFreeTools::compute_diagonal(
          this->matrix_free,
          vec_mass,
          &OperatorAllenCahnHelmholtz::do_vmult_cell_mass,
          this,
          this->dof_index);
      }

      {
        MyScope scope(this->timer, "helmholtz_op::get_diagonal::laplace");
        this->initialize_dof_vector(vec_laplace);
        MatrixFreeTools::compute_diagonal(
          this->matrix_free,
          vec_laplace,
          &OperatorAllenCahnHelmholtz::do_vmult_cell_laplace,
          this,
          this->dof_index);
      }
    }

  private:
    void
    do_vmult_cell_mass(FECellIntegrator &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate(EvaluationFlags::EvaluationFlags::values);
    }
    void
    do_vmult_cell_laplace(FECellIntegrator &phi) const
    {
      const auto &L       = this->op.get_data().L;
      const auto &kappa_p = this->op.get_data().kappa_p;

      phi.evaluate(EvaluationFlags::EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(L * kappa_p * phi.get_gradient(q), q);

      phi.integrate(EvaluationFlags::EvaluationFlags::gradients);
    }

    void
    do_vmult_kernel(FECellIntegrator &) const final
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;
  };



  template <int dim,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class InverseDiagonalMatrixAllenCahnHelmholtz
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using Operator   = OperatorAllenCahnHelmholtz<dim,
                                                n_components_,
                                                Number,
                                                VectorizedArrayType>;
    using VectorType = typename Operator::VectorType;

    static constexpr unsigned int n_components = n_components_ - 2;

    InverseDiagonalMatrixAllenCahnHelmholtz(const Operator &op)
      : op(op)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
      , dt(0.0)
    {}

    ~InverseDiagonalMatrixAllenCahnHelmholtz()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    virtual void
    clear() override
    {
      this->diag.reinit(0);
      this->vec_mass.reinit(0);
      this->vec_laplace.reinit(0);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      MyScope scope(this->timer, "helmholtz_precon::vmult");

      double *__restrict__ dst_ptr  = dst.get_values();
      double *__restrict__ src_ptr  = src.get_values();
      double *__restrict__ diag_ptr = diag.get_values();

      AssertDimension(n_components, 2);

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < diag.locally_owned_size(); ++i)
        {
          dst_ptr[i * 2 + 0] = diag_ptr[i] * src_ptr[i * 2 + 0];
          dst_ptr[i * 2 + 1] = diag_ptr[i] * src_ptr[i * 2 + 1];
        }
    }

    void
    do_update() override
    {
      MyScope scope(this->timer, "helmholtz_precon::do_update");

      const double new_dt = op.get_dt();

      if (diag.size() == 0)
        {
          op.initialize_dof_vector(diag);
          op.get_diagonals(vec_mass, vec_laplace);
        }

      if (this->dt != new_dt)
        {
          this->dt = new_dt;

          const double dt_inv = 1.0 / this->dt;

          for (unsigned int i = 0; i < diag.locally_owned_size(); ++i)
            {
              const double val = dt_inv * vec_mass.local_element(i) +
                                 vec_laplace.local_element(i);

              diag.local_element(i) =
                (std::abs(val) > 1.0e-10) ? (1.0 / val) : 1.0;
            }
        }
    }

  private:
    const Operator &op;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    double dt = 0.0;

    VectorType diag, vec_mass, vec_laplace;
  };



  struct BlockPreconditioner2Data
  {
    std::string block_0_preconditioner = "ILU";
    std::string block_1_preconditioner = "InverseDiagonalMatrix";
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class BlockPreconditioner2
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner2(
      const SinteringOperator<dim, n_components, Number, VectorizedArrayType>
        &                                                   op,
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const BlockPreconditioner2Data &                      data = {})
      : matrix_free(matrix_free)
      , operator_0(matrix_free, constraints, op)
      , operator_1(matrix_free, constraints, op)
      , operator_1_helmholtz(matrix_free, constraints, op)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
      , data(data)
    {
      preconditioner_0 =
        Preconditioners::create(operator_0, data.block_0_preconditioner);

      if (true /*TODO*/)
        preconditioner_1 =
          Preconditioners::create(operator_1, data.block_1_preconditioner);
      else
        preconditioner_1 = std::make_unique<
          InverseDiagonalMatrixAllenCahnHelmholtz<dim,
                                                  n_components,
                                                  Number,
                                                  VectorizedArrayType>>(
          operator_1_helmholtz);
    }

    ~BlockPreconditioner2()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    virtual void
    clear()
    {
      operator_0.clear();
      operator_1.clear();
      operator_1_helmholtz.clear();
      preconditioner_0->clear();
      preconditioner_1->clear();

      dst_0.reinit(0);
      src_0.reinit(0);
      dst_1.reinit(0);
      src_1.reinit(0);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      MyScope scope(timer, "precon::vmult");

      {
        MyScope scope(timer, "precon::vmult::split_up");
        VectorTools::split_up_fast(src, src_0, src_1);

#ifdef DEBUG
        VectorType temp_0, temp_1;
        temp_0.reinit(src_0);
        temp_1.reinit(src_1);

        VectorTools::split_up(this->matrix_free, src, temp_0, temp_1);

        AssertThrow(VectorTools::check_identity(src_0, temp_0),
                    ExcInternalError());
        AssertThrow(VectorTools::check_identity(src_1, temp_1),
                    ExcInternalError());
#endif
      }

      if (true)
        {
          MyScope scope(timer, "precon::vmult::precon_0");
          preconditioner_0->vmult(dst_0, src_0);
        }
      else
        {
          MyScope scope(timer, "precon::vmult::precon_0");

          try
            {
              ReductionControl reduction_control(100, 1e-20, 1e-8);

              SolverGMRES<VectorType> solver(reduction_control);
              solver.solve(operator_0, dst_0, src_0, *preconditioner_0);
            }
          catch (const SolverControl::NoConvergence &)
            {
              // TODO
            }
        }

      {
        MyScope scope(timer, "precon::vmult::precon_1");
        preconditioner_1->vmult(dst_1, src_1);
      }

      {
        MyScope scope(timer, "precon::vmult::merge");
        VectorTools::merge_fast(dst_0, dst_1, dst);

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    do_update() override
    {
      MyScope scope(timer, "precon::update");

      if (dst_0.size() == 0)
        {
          AssertDimension(src_0.size(), 0);
          AssertDimension(dst_1.size(), 0);
          AssertDimension(src_1.size(), 0);

          matrix_free.initialize_dof_vector(dst_0, 1);
          matrix_free.initialize_dof_vector(src_0, 1);
          matrix_free.initialize_dof_vector(dst_1, 2);
          matrix_free.initialize_dof_vector(src_1, 2);
        }

      {
        MyScope scope(timer, "precon::update::precon_0");
        preconditioner_0->do_update();
      }
      {
        MyScope scope(timer, "precon::update::precon_1");
        preconditioner_1->do_update();
      }
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    OperatorCahnHilliard<dim, 2, n_components, Number, VectorizedArrayType>
      operator_0;
    OperatorAllenCahn<dim,
                      n_components - 2,
                      n_components,
                      Number,
                      VectorizedArrayType>
      operator_1;
    OperatorAllenCahnHelmholtz<dim, n_components, Number, VectorizedArrayType>
      operator_1_helmholtz;

    mutable VectorType dst_0, dst_1;
    mutable VectorType src_0, src_1;

    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_1;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    const BlockPreconditioner2Data data;
  };



  struct BlockPreconditioner3Data
  {
    std::string type                       = "LD";
    std::string block_0_preconditioner     = "ILU";
    double      block_0_relative_tolerance = 0.0;
    std::string block_1_preconditioner     = "ILU";
    double      block_1_relative_tolerance = 0.0;
    std::string block_2_preconditioner     = "InverseDiagonalMatrix";
    double      block_2_relative_tolerance = 0.0;
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class BlockPreconditioner3
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner3(
      const SinteringOperator<dim, n_components, Number, VectorizedArrayType>
        &                                                   op,
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const BlockPreconditioner3Data &                      data = {})
      : matrix_free(matrix_free)
      , operator_0(matrix_free, constraints, op)
      , block_ch_b(matrix_free, constraints, op)
      , block_ch_c(matrix_free, constraints, op)
      , operator_1(matrix_free, constraints, op)
      , operator_2(matrix_free, constraints, op)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 1)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
      , data(data)
    {
      matrix_free.initialize_dof_vector(dst_0, 3 /*TODO*/);
      matrix_free.initialize_dof_vector(src_0, 3 /*TODO*/);

      matrix_free.initialize_dof_vector(dst_1, 3 /*TODO*/);
      matrix_free.initialize_dof_vector(src_1, 3 /*TODO*/);

      matrix_free.initialize_dof_vector(dst_2, 2);
      matrix_free.initialize_dof_vector(src_2, 2);

      preconditioner_0 =
        Preconditioners::create(operator_0, data.block_0_preconditioner);
      preconditioner_1 =
        Preconditioners::create(operator_1, data.block_1_preconditioner);
      preconditioner_2 =
        Preconditioners::create(operator_2, data.block_2_preconditioner);
    }

    ~BlockPreconditioner3()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      {
        MyScope scope(timer, "vmult::split_up");
        VectorTools::split_up_fast(src, src_0, src_1, src_2);

#ifdef DEBUG
        VectorType temp_0, temp_1, temp_2;
        temp_0.reinit(src_0);
        temp_1.reinit(src_1);
        temp_2.reinit(src_2);

        VectorTools::split_up(this->matrix_free, src, temp_0, temp_1, temp_2);

        AssertThrow(VectorTools::check_identity(src_0, temp_0),
                    ExcInternalError());
        AssertThrow(VectorTools::check_identity(src_1, temp_1),
                    ExcInternalError());
        AssertThrow(VectorTools::check_identity(src_2, temp_2),
                    ExcInternalError());
#endif
      }

      if (data.type == "D")
        {
          // Block Jacobi
          {
            MyScope scope(timer, "vmult::precon_0");

            if (data.block_0_relative_tolerance == 0.0)
              {
                preconditioner_0->vmult(dst_0, src_0);
              }
            else
              {
                ReductionControl reduction_control(
                  1000, 1e-20, data.block_0_relative_tolerance);

                SolverGMRES<VectorType> solver(reduction_control);
                solver.solve(operator_0, dst_0, src_0, *preconditioner_0);
              }
          }

          {
            MyScope scope(timer, "vmult::precon_1");

            if (data.block_1_relative_tolerance == 0.0)
              {
                preconditioner_1->vmult(dst_1, src_1);
              }
            else
              {
                ReductionControl reduction_control(
                  1000, 1e-20, data.block_1_relative_tolerance);

                SolverGMRES<VectorType> solver(reduction_control);
                solver.solve(operator_1, dst_1, src_1, *preconditioner_1);
              }
          }
        }
      else if (data.type == "LD")
        {
          // Block Gauss Seidel: L+D
          VectorType tmp;
          tmp.reinit(src_0);

          if (data.block_0_relative_tolerance == 0.0)
            {
              preconditioner_0->vmult(dst_0, src_0);
            }
          else
            {
              ReductionControl reduction_control(
                100, 1e-20, data.block_0_relative_tolerance);

              SolverGMRES<VectorType> solver(reduction_control);
              solver.solve(operator_0, dst_0, src_0, *preconditioner_0);
            }

          block_ch_c.vmult(tmp, dst_0);
          src_1 -= tmp;

          if (data.block_1_relative_tolerance == 0.0)
            {
              preconditioner_1->vmult(dst_1, src_1);
            }
          else
            {
              ReductionControl reduction_control(
                100, 1e-20, data.block_1_relative_tolerance);

              SolverGMRES<VectorType> solver(reduction_control);
              solver.solve(operator_1, dst_1, src_1, *preconditioner_1);
            }
        }
      else if (data.type == "RD")
        {
          // Block Gauss Seidel: R+D
          VectorType tmp;
          tmp.reinit(src_0);

          if (data.block_1_relative_tolerance == 0.0)
            {
              preconditioner_1->vmult(dst_1, src_1);
            }
          else
            {
              ReductionControl reduction_control(
                100, 1e-20, data.block_1_relative_tolerance);

              SolverGMRES<VectorType> solver(reduction_control);
              solver.solve(operator_1, dst_1, src_1, *preconditioner_1);
            }

          block_ch_b.vmult(tmp, dst_1);
          src_0 -= tmp;

          if (data.block_0_relative_tolerance == 0.0)
            {
              preconditioner_0->vmult(dst_0, src_0);
            }
          else
            {
              ReductionControl reduction_control(
                100, 1e-20, data.block_0_relative_tolerance);

              SolverGMRES<VectorType> solver(reduction_control);
              solver.solve(operator_0, dst_0, src_0, *preconditioner_0);
            }
        }
      else if (data.type == "SYMM")
        {
          // Block Gauss Seidel: symmetric
          AssertThrow(false, ExcNotImplemented());
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      {
        AssertThrow(data.block_2_relative_tolerance == 0.0,
                    ExcNotImplemented());

        MyScope scope(timer, "vmult::precon_2");
        preconditioner_2->vmult(dst_2, src_2);
      }

      {
        MyScope scope(timer, "vmult::merge");
        VectorTools::merge_fast(dst_0, dst_1, dst_2, dst);

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, dst_2, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    do_update() override
    {
      preconditioner_0->do_update();
      preconditioner_1->do_update();
      preconditioner_2->do_update();
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    OperatorCahnHilliardA<dim, 1, n_components, Number, VectorizedArrayType>
      operator_0;
    OperatorCahnHilliardB<dim, 1, n_components, Number, VectorizedArrayType>
      block_ch_b;
    OperatorCahnHilliardC<dim, 1, n_components, Number, VectorizedArrayType>
      block_ch_c;
    OperatorCahnHilliardD<dim, 1, n_components, Number, VectorizedArrayType>
      operator_1;

    OperatorAllenCahn<dim,
                      n_components - 2,
                      n_components,
                      Number,
                      VectorizedArrayType>
      operator_2;

    mutable VectorType dst_0, dst_1, dst_2;
    mutable VectorType src_0, src_1, src_2;

    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_1, preconditioner_2;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    BlockPreconditioner3Data data;
  };


  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class MassMatrix
    : public OperatorBase<dim, n_components, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    MassMatrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints)
      : OperatorBase<dim, n_components, Number, VectorizedArrayType>(
          matrix_free,
          constraints,
          3,
          "mass_matrix_op")
    {}

  private:
    void
    do_vmult_kernel(FECellIntegrator &phi) const final
    {
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient({}, q);
        }
    }
  };



  template <int dim,
            int n_components_,
            typename Number,
            typename VectorizedArrayType>
  class OperatorCahnHilliardHelmholtz
    : public OperatorBase<dim, 1, Number, VectorizedArrayType>
  {
  public:
    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>;

    using VectorType =
      typename OperatorBase<dim, 1, Number, VectorizedArrayType>::VectorType;

    OperatorCahnHilliardHelmholtz(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
        &op)
      : OperatorBase<dim, 1, Number, VectorizedArrayType>(matrix_free,
                                                          constraints,
                                                          3,
                                                          "ch_helmholtz_op")
      , op(op)
      , dt(0.0)
    {}

    double
    get_dt() const
    {
      return op.get_dt();
    }

    double
    get_sqrt_delta() const
    {
      return std::sqrt(this->op.get_data().kappa_c);
    }

    const VectorType &
    get_epsilon() const
    {
      const double new_dt = op.get_dt();

      if (epsilon.size() == 0)
        {
          this->initialize_dof_vector(epsilon);
        }

      if (this->dt != new_dt)
        {
          this->dt = new_dt;

          VectorType vec_w_mobility, vec_wo_mobility;

          this->initialize_dof_vector(vec_w_mobility);
          this->initialize_dof_vector(vec_wo_mobility);

          MatrixFreeTools::compute_diagonal(
            this->matrix_free,
            vec_w_mobility,
            &OperatorCahnHilliardHelmholtz::do_vmult_cell_laplace<true>,
            this,
            this->dof_index);

          MatrixFreeTools::compute_diagonal(
            this->matrix_free,
            vec_wo_mobility,
            &OperatorCahnHilliardHelmholtz::do_vmult_cell_laplace<false>,
            this,
            this->dof_index);

          for (unsigned int i = 0; i < epsilon.locally_owned_size(); ++i)
            epsilon.local_element(i) = vec_w_mobility.local_element(i) /
                                       vec_wo_mobility.local_element(i) *
                                       std::sqrt(dt);

          if (true /*TODO*/)
            {
              // perfom limiting
              const auto max_value = [this]() {
                typename VectorType::value_type temp = 0;

                for (const auto i : epsilon)
                  temp = std::max(temp, i);

                temp = Utilities::MPI::max(temp, MPI_COMM_WORLD);

                return temp;
              }();

              for (auto &i : epsilon)
                i = std::max(i,
                             max_value /
                               100); // bound smallest entries by the max value
            }
        }

      return epsilon;
    }

  private:
    void
    do_vmult_kernel(FECellIntegrator &phi) const final
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      const auto sqrt_delta = this->get_sqrt_delta();
      const auto dt         = get_dt();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &c    = nonlinear_values(cell, q)[0];
          const auto &eta1 = nonlinear_values(cell, q)[2];
          const auto &eta2 = nonlinear_values(cell, q)[3];

          const auto &c_grad    = nonlinear_gradients(cell, q)[0];
          const auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          const auto &eta2_grad = nonlinear_gradients(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);
          const auto epsilon  = dt * mobility.M(c, etas, c_grad, etas_grad);

          phi.submit_value(value, q);
          phi.submit_gradient(std::sqrt(std::abs(sqrt_delta * epsilon)) *
                                gradient,
                              q);
        }
    }

    template <bool use_mobility>
    void
    do_vmult_cell_laplace(FECellIntegrator &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::gradients);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      const auto sqrt_delta = this->get_sqrt_delta();
      const auto dt         = get_dt();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &c    = nonlinear_values(cell, q)[0];
          const auto &eta1 = nonlinear_values(cell, q)[2];
          const auto &eta2 = nonlinear_values(cell, q)[3];

          const auto &c_grad    = nonlinear_gradients(cell, q)[0];
          const auto &eta1_grad = nonlinear_gradients(cell, q)[2];
          const auto &eta2_grad = nonlinear_gradients(cell, q)[3];

          const std::array<VectorizedArrayType, 2> etas{{eta1, eta2}};
          const std::array<Tensor<1, dim, VectorizedArrayType>, 2> etas_grad{
            {eta1_grad, eta2_grad}};

          const auto gradient = phi.get_gradient(q);
          const auto epsilon =
            dt * (use_mobility ? mobility.M(c, etas, c_grad, etas_grad) :
                                 VectorizedArrayType(1.0));

          phi.submit_gradient(std::sqrt(std::abs(sqrt_delta * epsilon)) *
                                gradient,
                              q);
        }

      phi.integrate(EvaluationFlags::EvaluationFlags::gradients);
    }

    const SinteringOperator<dim, n_components_, Number, VectorizedArrayType>
      &op;

    mutable VectorType epsilon;

    mutable double dt;
  };



  struct BlockPreconditioner3CHData
  {
    std::string block_0_preconditioner = "AMG";
    std::string block_2_preconditioner = "InverseDiagonalMatrix";
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class BlockPreconditioner3CHOperator : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    BlockPreconditioner3CHOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const SinteringOperator<dim, n_components, Number, VectorizedArrayType>
        &op)
      : operator_a(matrix_free, constraints, op)
      , operator_b(matrix_free, constraints, op)
      , operator_c(matrix_free, constraints, op)
      , operator_d(matrix_free, constraints, op)
    {}

    void
    vmult(LinearAlgebra::distributed::BlockVector<Number> &      dst,
          const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      VectorType temp;
      temp.reinit(src.block(0));

      operator_a.vmult(dst.block(0), src.block(0));
      operator_b.vmult(temp, src.block(1));
      dst.block(0).add(1.0, temp);

      operator_c.vmult(dst.block(1), src.block(0));
      operator_d.vmult(temp, src.block(1));
      dst.block(1).add(1.0, temp);
    }

  private:
    OperatorCahnHilliardA<dim, 1, n_components, Number, VectorizedArrayType>
      operator_a;
    OperatorCahnHilliardB<dim, 1, n_components, Number, VectorizedArrayType>
      operator_b;
    OperatorCahnHilliardC<dim, 1, n_components, Number, VectorizedArrayType>
      operator_c;
    OperatorCahnHilliardD<dim, 1, n_components, Number, VectorizedArrayType>
      operator_d;
  };



  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class BlockPreconditioner3CHPreconditioner
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;


    BlockPreconditioner3CHPreconditioner(
      const OperatorCahnHilliardHelmholtz<dim,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &operator_0,
      const MassMatrix<dim, 1, Number, VectorizedArrayType> &   mass_matrix,
      const std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        &preconditioner_0)
      : operator_0(operator_0)
      , mass_matrix(mass_matrix)
      , preconditioner_0(preconditioner_0)
    {}

    void
    vmult(LinearAlgebra::distributed::BlockVector<Number> &      dst,
          const LinearAlgebra::distributed::BlockVector<Number> &src) const
    {
      const auto &src_0 = src.block(0);
      const auto &src_1 = src.block(1);
      auto &      dst_0 = dst.block(0);
      auto &      dst_1 = dst.block(1);

      VectorType b_0, b_1, g; // TODO: reduce number of temporal vectors
      b_0.reinit(src_0);      //
      b_1.reinit(src_0);      //
      g.reinit(src_0);        //

      const auto &epsilon    = operator_0.get_epsilon();
      const auto  sqrt_delta = operator_0.get_sqrt_delta();
      const auto  dt         = operator_0.get_dt();

      // b_0
      for (unsigned int i = 0; i < src_0.locally_owned_size(); ++i)
        b_0.local_element(i) = sqrt_delta / epsilon.local_element(i) *
                                 (dt * src_0.local_element(i)) +
                               src_1.local_element(i);

      // g
      preconditioner_0->vmult(g, b_0);

      // b_1
      mass_matrix.vmult(b_1, g);
      for (unsigned int i = 0; i < src_0.locally_owned_size(); ++i)
        b_1.local_element(i) -=
          sqrt_delta / epsilon.local_element(i) * (dt * src_0.local_element(i));

      // x_0 tilde
      preconditioner_0->vmult(dst_1, b_1);

      // x_0 and x_1
      for (unsigned int i = 0; i < src_0.locally_owned_size(); ++i)
        {
          dst_0.local_element(i) =
            epsilon.local_element(i) / sqrt_delta *
            (g.local_element(i) - dst_1.local_element(i));
          dst_1.local_element(i) *= -1.0;
        }
    }

  private:
    const OperatorCahnHilliardHelmholtz<dim,
                                        n_components,
                                        Number,
                                        VectorizedArrayType> &operator_0;

    const MassMatrix<dim, 1, Number, VectorizedArrayType> &mass_matrix;

    const std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      &preconditioner_0;
  };

  template <int dim,
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class BlockPreconditioner3CH
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner3CH(
      const SinteringOperator<dim, n_components, Number, VectorizedArrayType>
        &                                                   op,
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const BlockPreconditioner3CHData &                    data = {})
      : matrix_free(matrix_free)
      , operator_0(matrix_free, constraints, op)
      , mass_matrix(matrix_free, constraints)
      , operator_2(matrix_free, constraints, op)
      , op_ch(matrix_free, constraints, op)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 1)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
      , data(data)
    {
      matrix_free.initialize_dof_vector(dst_0, 3 /*TODO*/);
      matrix_free.initialize_dof_vector(src_0, 3 /*TODO*/);

      matrix_free.initialize_dof_vector(dst_1, 3 /*TODO*/);
      matrix_free.initialize_dof_vector(src_1, 3 /*TODO*/);

      matrix_free.initialize_dof_vector(dst_2, 2);
      matrix_free.initialize_dof_vector(src_2, 2);

      preconditioner_0 =
        Preconditioners::create(operator_0, data.block_0_preconditioner);
      preconditioner_2 =
        Preconditioners::create(operator_2, data.block_2_preconditioner);
    }

    ~BlockPreconditioner3CH()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const override
    {
      {
        MyScope scope(timer, "vmult::split_up");
        VectorTools::split_up_fast(src, src_0, src_1, src_2);

#ifdef DEBUG
        VectorType temp_0, temp_1, temp_2;
        temp_0.reinit(src_0);
        temp_1.reinit(src_1);
        temp_2.reinit(src_2);

        VectorTools::split_up(this->matrix_free, src, temp_0, temp_1, temp_2);

        AssertThrow(VectorTools::check_identity(src_0, temp_0),
                    ExcInternalError());
        AssertThrow(VectorTools::check_identity(src_1, temp_1),
                    ExcInternalError());
        AssertThrow(VectorTools::check_identity(src_2, temp_2),
                    ExcInternalError());
#endif
      }

      {
        LinearAlgebra::distributed::BlockVector<Number> src_block(2);
        LinearAlgebra::distributed::BlockVector<Number> dst_block(2);

        src_block.block(0) = src_0;
        src_block.block(1) = src_1;

        dst_block.block(0).reinit(dst_0);
        dst_block.block(1).reinit(dst_1);

        auto precon_inner = std::make_shared<
          BlockPreconditioner3CHPreconditioner<dim,
                                               n_components,
                                               Number,
                                               VectorizedArrayType>>(
          operator_0, mass_matrix, preconditioner_0);

        if (false)
          {
            precon_inner->vmult(dst_block, src_block);
          }
        else if (true)
          {
            using RelaxationType = PreconditionRelaxation<
              BlockPreconditioner3CHOperator<dim,
                                             n_components,
                                             Number,
                                             VectorizedArrayType>,
              BlockPreconditioner3CHPreconditioner<dim,
                                                   n_components,
                                                   Number,
                                                   VectorizedArrayType>>;

            typename RelaxationType::AdditionalData ad;

            ad.preconditioner = precon_inner;
            ad.n_iterations   = 1;
            ad.relaxation     = 1.0;

            RelaxationType precon;
            precon.initialize(op_ch, ad);
            precon.vmult(dst_block, src_block);
          }
        else
          {
            try
              {
                ReductionControl reduction_control(10, 1e-20, 1e-4);

                SolverGMRES<LinearAlgebra::distributed::BlockVector<Number>>
                  solver(reduction_control);
                solver.solve(op_ch, dst_block, src_block, *precon_inner);
              }
            catch (const SolverControl::NoConvergence &)
              {
                // TODO
              }
          }

        dst_0 = dst_block.block(0);
        dst_1 = dst_block.block(1);
      }

      {
        MyScope scope(timer, "vmult::precon_2");
        preconditioner_2->vmult(dst_2, src_2);
      }

      {
        MyScope scope(timer, "vmult::merge");
        VectorTools::merge_fast(dst_0, dst_1, dst_2, dst);

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, dst_2, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    do_update() override
    {
      preconditioner_0->do_update();
      preconditioner_2->do_update();
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    OperatorCahnHilliardHelmholtz<dim,
                                  n_components,
                                  Number,
                                  VectorizedArrayType>
      operator_0;

    MassMatrix<dim, 1, Number, VectorizedArrayType> mass_matrix;

    OperatorAllenCahn<dim,
                      n_components - 2,
                      n_components,
                      Number,
                      VectorizedArrayType>
      operator_2;


    const BlockPreconditioner3CHOperator<dim,
                                         n_components,
                                         Number,
                                         VectorizedArrayType>
      op_ch;

    mutable VectorType dst_0, dst_1, dst_2;
    mutable VectorType src_0, src_1, src_2;

    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_2;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    BlockPreconditioner3CHData data;
  };



  struct Parameters
  {
    unsigned int fe_degree   = 1;
    unsigned int n_points_1D = 2;

    double   top_fraction_of_cells    = 0.3;
    double   bottom_fraction_of_cells = 0.03;
    unsigned min_refinement_depth     = 2;
    unsigned max_refinement_depth     = 0;
    unsigned refinement_frequency     = 10;

    bool matrix_based = false;

    std::string outer_preconditioner = "BlockPreconditioner2";
    // std::string outer_preconditioner = "BlockPreconditioner3CH";

    BlockPreconditioner2Data   block_preconditioner_2_data;
    BlockPreconditioner3Data   block_preconditioner_3_data;
    BlockPreconditioner3CHData block_preconditioner_3_ch_data;

    bool print_time_loop = true;

    void
    parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
    }

    void
    print()
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      ConditionalOStream pcout(
        std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      if (pcout.is_active())
        prm.print_parameters(
          pcout.get_stream(),
          ParameterHandler::OutputStyle::Description |
            ParameterHandler::OutputStyle::KeepDeclarationOrder);
    }

  private:
    void
    add_parameters(ParameterHandler &prm)
    {
      const std::string preconditioner_types =
        "AMG|InverseBlockDiagonalMatrix|InverseDiagonalMatrix|ILU";

      prm.add_parameter("FEDegree",
                        fe_degree,
                        "Degree of the shape the finite element.");
      prm.add_parameter("NPoints1D",
                        n_points_1D,
                        "Number of quadrature points.");
      prm.add_parameter(
        "OuterPreconditioner",
        outer_preconditioner,
        "Preconditioner to be used for the outer system.",
        Patterns::Selection(
          preconditioner_types +
          "|BlockPreconditioner2|BlockPreconditioner3|BlockPreconditioner3CH"));

      prm.enter_subsection("BlockPreconditioner2");
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_2_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block1Preconditioner",
                        block_preconditioner_2_data.block_1_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3");
      prm.add_parameter("Type",
                        block_preconditioner_3_data.type,
                        "Type of block preconditioner of CH system.",
                        Patterns::Selection("D|LD|RD|SYMM"));
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_3_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block0RelativeTolerance",
                        block_preconditioner_3_data.block_0_relative_tolerance,
                        "Relative tolerance of the first block.");
      prm.add_parameter("Block1Preconditioner",
                        block_preconditioner_3_data.block_1_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block1RelativeTolerance",
                        block_preconditioner_3_data.block_1_relative_tolerance,
                        "Relative tolerance of the second block.");
      prm.add_parameter("Block2Preconditioner",
                        block_preconditioner_3_data.block_2_preconditioner,
                        "Preconditioner to be used for the thrird block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2RelativeTolerance",
                        block_preconditioner_3_data.block_2_relative_tolerance,
                        "Relative tolerance of the third block.");
      prm.leave_subsection();

      prm.enter_subsection("BlockPreconditioner3CH");
      prm.add_parameter("Block0Preconditioner",
                        block_preconditioner_3_ch_data.block_0_preconditioner,
                        "Preconditioner to be used for the first block.",
                        Patterns::Selection(preconditioner_types));
      prm.add_parameter("Block2Preconditioner",
                        block_preconditioner_3_ch_data.block_2_preconditioner,
                        "Preconditioner to be used for the second block.",
                        Patterns::Selection(preconditioner_types));
      prm.leave_subsection();
    }
  };



  template <int dim,
            typename Number              = double,
            typename VectorizedArrayType = VectorizedArray<Number>>
  class Problem
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    // components number
    static constexpr unsigned int number_of_components = 4;

    using NonLinearOperator =
      SinteringOperator<dim, number_of_components, Number, VectorizedArrayType>;

    // geometry
    static constexpr double diameter        = 15.0;
    static constexpr double interface_width = 2.0;
    static constexpr double boundary_factor = 1.0;

    // mesh
    static constexpr unsigned int elements_per_interface = 4;

    // time discretization
    static constexpr double t_end                = 100;
    static constexpr double dt_deseride          = 0.001;
    static constexpr double dt_max               = 1e3 * dt_deseride;
    static constexpr double dt_min               = 1e-2 * dt_deseride;
    static constexpr double dt_increment         = 1.2;
    static constexpr double output_time_interval = 1.0; // 0.0 means no output

    // desirable number of newton iterations
    static constexpr unsigned int desirable_newton_iterations = 5;
    static constexpr unsigned int desirable_linear_iterations = 100;

    //  model parameters
    static constexpr double A       = 16;
    static constexpr double B       = 1;
    static constexpr double Mvol    = 1e-2;
    static constexpr double Mvap    = 1e-10;
    static constexpr double Msurf   = 4;
    static constexpr double Mgb     = 0.4;
    static constexpr double L       = 1;
    static constexpr double kappa_c = 1;
    static constexpr double kappa_p = 0.5;

    // Create mesh
    static constexpr double domain_width =
      2 * diameter + boundary_factor * diameter;
    static constexpr double domain_height =
      1 * diameter + boundary_factor * diameter;

    static constexpr double x01             = domain_width / 2. - diameter / 2.;
    static constexpr double x02             = domain_width / 2. + diameter / 2.;
    static constexpr double y0              = domain_height / 2.;
    static constexpr double r0              = diameter / 2.;
    static constexpr bool   is_accumulative = false;

    const Parameters                          params;
    ConditionalOStream                        pcout;
    ConditionalOStream                        pcout_statistics;
    parallel::distributed::Triangulation<dim> tria;
    FESystem<dim>                             fe;
    MappingQ<dim>                             mapping;
    QGauss<dim>                               quad;
    DoFHandler<dim>                           dof_handler;
    DoFHandler<dim>                           dof_handler_ch;
    DoFHandler<dim>                           dof_handler_ac;
    DoFHandler<dim>                           dof_handler_scalar;

    AffineConstraints<Number> constraint;
    AffineConstraints<Number> constraint_ch;
    AffineConstraints<Number> constraint_ac;
    AffineConstraints<Number> constraint_scalar;

    const std::vector<const AffineConstraints<double> *> constraints;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    InitialValues<dim> initial_solution;

    Problem(const Parameters &params)
      : params(params)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                params.print_time_loop)
      , pcout_statistics(std::cout,
                         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , fe(FE_Q<dim>{params.fe_degree}, number_of_components)
      , mapping(1)
      , quad(params.n_points_1D)
      , dof_handler(tria)
      , dof_handler_ch(tria)
      , dof_handler_ac(tria)
      , dof_handler_scalar(tria)
      , constraints{&constraint,
                    &constraint_ch,
                    &constraint_ac,
                    &constraint_scalar}
      , initial_solution(x01,
                         x02,
                         y0,
                         r0,
                         interface_width,
                         number_of_components,
                         is_accumulative)
    {
      create_mesh(tria,
                  domain_width,
                  domain_height,
                  interface_width,
                  elements_per_interface);

      initialize();
    }

    void
    initialize()
    {
      // setup DoFHandler, ...
      dof_handler.distribute_dofs(fe);
      dof_handler_ch.distribute_dofs(
        FESystem<dim>(FE_Q<dim>{params.fe_degree}, 2));
      dof_handler_ac.distribute_dofs(
        FESystem<dim>(FE_Q<dim>{params.fe_degree}, number_of_components - 2));
      dof_handler_scalar.distribute_dofs(FE_Q<dim>{params.fe_degree});

      // ... constraints, and ...
      constraint.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);
      constraint.close();

      constraint_ch.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_ch, constraint_ch);
      constraint_ch.close();

      constraint_ac.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_ac, constraint_ac);
      constraint_ac.close();

      constraint_scalar.clear();
      DoFTools::make_hanging_node_constraints(dof_handler_scalar,
                                              constraint_scalar);
      constraint_scalar.close();

      // ... MatrixFree
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;
      // additional_data.use_fast_hanging_node_algorithm = false; // TODO

      const std::vector<const DoFHandler<dim> *> dof_handlers{
        &dof_handler, &dof_handler_ch, &dof_handler_ac, &dof_handler_scalar};

      matrix_free.reinit(
        mapping, dof_handlers, constraints, quad, additional_data);

      // clang-format off
      pcout_statistics << "System statistics:" << std::endl;
      pcout_statistics << "  - n cell:                    " << tria.n_global_active_cells() << std::endl;
      pcout_statistics << "  - n levels:                  " << tria.n_global_levels() << std::endl;
      pcout_statistics << "  - n dofs:                    " << dof_handler.n_dofs() << std::endl;
      pcout_statistics << std::endl;
      // clang-format on
    }

    void
    run()
    {
      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        A, B, Mvol, Mvap, Msurf, Mgb, L, kappa_c, kappa_p);

      // ... non-linear operator
      NonLinearOperator nonlinear_operator(matrix_free,
                                           constraints,
                                           sintering_data,
                                           params.matrix_based);

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations;
      MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        mg_matrixfrees;

      if (params.outer_preconditioner == "BlockPreconditioner2")
        preconditioner =
          std::make_unique<BlockPreconditioner2<dim,
                                                number_of_components,
                                                Number,
                                                VectorizedArrayType>>(
            nonlinear_operator,
            matrix_free,
            constraints,
            params.block_preconditioner_2_data);
      else if (params.outer_preconditioner == "BlockPreconditioner3")
        preconditioner =
          std::make_unique<BlockPreconditioner3<dim,
                                                number_of_components,
                                                Number,
                                                VectorizedArrayType>>(
            nonlinear_operator,
            matrix_free,
            constraints,
            params.block_preconditioner_3_data);
      else if (params.outer_preconditioner == "BlockPreconditioner3CH")
        preconditioner =
          std::make_unique<BlockPreconditioner3CH<dim,
                                                  number_of_components,
                                                  Number,
                                                  VectorizedArrayType>>(
            nonlinear_operator,
            matrix_free,
            constraints,
            params.block_preconditioner_3_ch_data);
      else
        preconditioner = Preconditioners::create(nonlinear_operator,
                                                 params.outer_preconditioner);

      // ... linear solver
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (true)
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearOperator,
          Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                        *preconditioner);

      TimerOutput timer(pcout_statistics,
                        TimerOutput::never,
                        TimerOutput::wall_times);

      // ... non-linear Newton solver
      auto non_linear_solver =
        std::make_unique<NonLinearSolvers::NewtonSolver<VectorType>>();

      non_linear_solver->reinit_vector = [&](auto &vector) {
        MyScope scope(timer, "time_loop::newton::reinit_vector");

        nonlinear_operator.initialize_dof_vector(vector);
      };

      non_linear_solver->residual = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::residual");

        nonlinear_operator.evaluate_nonlinear_residual(dst, src);
      };

      non_linear_solver->setup_jacobian =
        [&](const auto &current_u, const bool do_update_preconditioner) {
          if (true)
            {
              MyScope scope(timer, "time_loop::newton::setup_jacobian");

              nonlinear_operator.evaluate_newton_step(current_u);
            }

          if (do_update_preconditioner)
            {
              MyScope scope(timer, "time_loop::newton::setup_preconditioner");
              preconditioner->do_update();
            }
        };

      non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
        MyScope scope(timer, "time_loop::newton::solve_with_jacobian");

        // note: we mess with the input here, since we know that Newton does not
        // use the content anymore
        constraint.set_zero(const_cast<VectorType &>(src));
        const unsigned int n_iterations = linear_solver->solve(dst, src);
        constraint.distribute(dst);

        return n_iterations;
      };


      // set initial condition
      VectorType solution;

      nonlinear_operator.initialize_dof_vector(solution);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               initial_solution,
                               solution);
      solution.zero_out_ghost_values();

      double time_last_output = 0;

      if (output_time_interval > 0.0)
        output_result(solution, nonlinear_operator, time_last_output);

      unsigned int n_timestep              = 0;
      unsigned int n_linear_iterations     = 0;
      unsigned int n_non_linear_iterations = 0;
      double       max_reached_dt          = 0.0;

      const unsigned int init_level = tria.n_global_levels() - 1;

      // run time loop
      {
        TimerOutput::Scope scope(timer, "time_loop");
        for (double t = 0, dt = dt_deseride; t <= t_end;)
          {
            if (n_timestep != 0 && params.refinement_frequency > 0 &&
                n_timestep % params.refinement_frequency == 0)
              {
                pcout << "Execute refinement/coarsening:" << std::endl;

                // 1) copy solution so that it has the right ghosting
                IndexSet locally_relevant_dofs;
                DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                        locally_relevant_dofs);
                VectorType solution_dealii(dof_handler.locally_owned_dofs(),
                                           locally_relevant_dofs,
                                           dof_handler.get_communicator());

                // note: we do not need to apply constraints, since they are
                // are already set by the Newton solver
                solution_dealii.copy_locally_owned_data_from(solution);
                solution_dealii.update_ghost_values();

                // 2) estimate errors
                Vector<float> estimated_error_per_cell(tria.n_active_cells());

                std::vector<bool> mask(number_of_components, true);
                std::fill(mask.begin(), mask.begin() + 2, false);

                KellyErrorEstimator<dim>::estimate(
                  this->dof_handler,
                  QGauss<dim - 1>(this->dof_handler.get_fe().degree + 1),
                  std::map<types::boundary_id, const Function<dim> *>(),
                  solution_dealii,
                  estimated_error_per_cell,
                  mask,
                  nullptr,
                  0,
                  Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

                // 3) mark cells
                parallel::distributed::GridRefinement::
                  refine_and_coarsen_fixed_fraction(
                    tria,
                    estimated_error_per_cell,
                    params.top_fraction_of_cells,
                    params.bottom_fraction_of_cells);

                if (tria.n_levels() > init_level + params.max_refinement_depth)
                  for (const auto &cell : tria.active_cell_iterators_on_level(
                         init_level + params.max_refinement_depth))
                    cell->clear_refine_flag();

                for (const auto &cell : tria.active_cell_iterators_on_level(
                       init_level - params.min_refinement_depth))
                  cell->clear_coarsen_flag();

                // 4) perform interpolation and initialize data structures
                tria.prepare_coarsening_and_refinement();

                parallel::distributed::SolutionTransfer<dim, VectorType>
                  solution_trans(dof_handler);
                solution_trans.prepare_for_coarsening_and_refinement(
                  solution_dealii);

                tria.execute_coarsening_and_refinement();

                initialize();

                nonlinear_operator.clear();
                preconditioner->clear();

                VectorType interpolated_solution;
                nonlinear_operator.initialize_dof_vector(interpolated_solution);
                solution_trans.interpolate(interpolated_solution);

                nonlinear_operator.initialize_dof_vector(solution);
                solution.copy_locally_owned_data_from(interpolated_solution);

                // note: apply constraints since the Newton solver expects this
                constraint.distribute(solution);
              }

            nonlinear_operator.set_timestep(dt);
            nonlinear_operator.set_previous_solution(solution);

            bool has_converged = false;

            try
              {
                MyScope scope(timer, "time_loop::newton");

                // note: input/output (solution) needs/has the right
                // constraints applied
                const auto statistics = non_linear_solver->solve(solution);

                has_converged = true;

                pcout << "t = " << t << ", dt = " << dt << ":"
                      << " solved in " << statistics.newton_iterations
                      << " Newton iterations and "
                      << statistics.linear_iterations << " linear iterations"
                      << std::endl;

                n_timestep += 1;
                n_linear_iterations += statistics.linear_iterations;
                n_non_linear_iterations += statistics.newton_iterations;
                max_reached_dt = std::max(max_reached_dt, dt);

                if (std::abs(t - t_end) > 1e-9)
                  {
                    if (statistics.newton_iterations <
                          desirable_newton_iterations &&
                        statistics.linear_iterations <
                          desirable_linear_iterations)
                      {
                        dt *= dt_increment;
                        pcout << "\033[32mIncreasing timestep, dt = " << dt
                              << "\033[0m" << std::endl;

                        if (dt > dt_max)
                          {
                            dt = dt_max;
                          }
                      }

                    if (t + dt > t_end)
                      {
                        dt = t_end - t;
                      }
                  }

                t += dt;
              }
            catch (const NonLinearSolvers::ExcNewtonDidNotConverge &)
              {
                dt *= 0.5;
                pcout << "\033[31mSolver diverged, reducing timestep, dt = "
                      << dt << "\033[0m" << std::endl;

                solution = nonlinear_operator.get_previous_solution();

                AssertThrow(
                  dt > dt_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }
            catch (const SolverControl::NoConvergence &)
              {
                dt *= 0.5;
                pcout << "\033[33mSolver diverged, reducing timestep, dt = "
                      << dt << "\033[0m" << std::endl;

                solution = nonlinear_operator.get_previous_solution();

                AssertThrow(
                  dt > dt_min,
                  ExcMessage(
                    "Minimum timestep size exceeded, solution failed!"));
              }

            if ((output_time_interval > 0.0) && has_converged &&
                (t > output_time_interval + time_last_output))
              {
                time_last_output = t;
                output_result(solution, nonlinear_operator, time_last_output);
              }
          }
      }

      // clang-format off
      pcout_statistics << std::endl;
      pcout_statistics << "Final statistics:" << std::endl;
      pcout_statistics << "  - n timesteps:               " << n_timestep << std::endl;
      pcout_statistics << "  - n non-linear iterations:   " << n_non_linear_iterations << std::endl;
      pcout_statistics << "  - n linear iterations:       " << n_linear_iterations << std::endl;
      pcout_statistics << "  - avg non-linear iterations: " << static_cast<double>(n_non_linear_iterations) / n_timestep << std::endl;
      pcout_statistics << "  - avg linear iterations:     " << static_cast<double>(n_linear_iterations) / n_non_linear_iterations << std::endl;
      pcout_statistics << "  - max dt:                    " << max_reached_dt << std::endl;
      pcout_statistics << std::endl;
      // clang-format on

      timer.print_wall_time_statistics(MPI_COMM_WORLD);

      {
        nonlinear_operator.set_timestep(dt_deseride);
        nonlinear_operator.set_previous_solution(solution);
        nonlinear_operator.evaluate_newton_step(solution);

        VectorType dst, src;

        nonlinear_operator.initialize_dof_vector(dst);
        nonlinear_operator.initialize_dof_vector(src);

        const unsigned int n_repetitions = 1000;

        TimerOutput timer(pcout_statistics,
                          TimerOutput::never,
                          TimerOutput::wall_times);

        {
          TimerOutput::Scope scope(timer, "vmult_matrixfree");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            nonlinear_operator.vmult(dst, src);
        }

        {
          const auto &matrix = nonlinear_operator.get_system_matrix();

          TimerOutput::Scope scope(timer, "vmult_matrixbased");

          for (unsigned int i = 0; i < n_repetitions; ++i)
            matrix.vmult(dst, src);
        }

        timer.print_wall_time_statistics(MPI_COMM_WORLD);
      }
    }

  private:
    void
    create_mesh(parallel::distributed::Triangulation<dim> &tria,
                const double                               domain_width,
                const double                               domain_height,
                const double                               interface_width,
                const unsigned int elements_per_interface)
    {
      const unsigned int initial_ny = 50;
      const unsigned int initial_nx =
        int(domain_width / domain_height * initial_ny);

      const unsigned int n_refinements =
        int(std::round(std::log2(elements_per_interface / interface_width *
                                 domain_height / initial_ny)));

      std::vector<unsigned int> subdivisions(dim);
      subdivisions[0] = initial_nx;
      subdivisions[1] = initial_ny;
      if (dim == 3)
        subdivisions[2] = initial_ny;

      const dealii::Point<dim> bottom_left;
      const dealii::Point<dim> top_right =
        (dim == 2 ?
           dealii::Point<dim>(domain_width, domain_height) :
           dealii::Point<dim>(domain_width, domain_height, domain_height));

      dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                        subdivisions,
                                                        bottom_left,
                                                        top_right);

      if (n_refinements > 0)
        tria.refine_global(n_refinements);
    }

    void
    output_result(const VectorType &       solution,
                  const NonLinearOperator &sintering_operator,
                  const double             t)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      std::vector<std::string> names{"c", "mu", "eta1", "eta2"};
      data_out.add_data_vector(solution, names);

      sintering_operator.add_data_vectors(data_out, solution);

      data_out.build_patches(mapping, this->fe.tensor_degree());

      static unsigned int counter = 0;

      pcout << "Outputing at t = " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };
  };
} // namespace Sintering



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  Sintering::Parameters params;

  if (argc == 2)
    {
      if (std::string(argv[1]) == "--help")
        {
          params.print();
          return 0;
        }
      else
        {
          params.parse(std::string(argv[1]));
        }
    }

  Sintering::Problem<SINTERING_DIM> runner(params);
  runner.run();
}
