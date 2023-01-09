#pragma once

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

namespace dealii
{
  namespace MyMatrixFreeTools
  {
    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints_in,
      TrilinosWrappers::SparseMatrix &                    matrix,
      const std::function<void(FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components,
                                            Number,
                                            VectorizedArrayType> &)>
        &                local_vmult,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0)
    {
      using MatrixType = TrilinosWrappers::SparseMatrix;

      // new
      AssertDimension(
        matrix_free.get_dof_handler(dof_no).get_fe().n_components(), 1);

      std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
        constraints_for_matrix;
      const AffineConstraints<typename MatrixType::value_type> &constraints =
        dealii::MatrixFreeTools::internal::
          create_new_affine_constraints_if_needed(matrix,
                                                  constraints_in,
                                                  constraints_for_matrix);

      matrix_free.template cell_loop<MatrixType, MatrixType>(
        [&](const auto &, auto &dst, const auto &, const auto range) {
          FEEvaluation<dim,
                       fe_degree,
                       n_q_points_1d,
                       n_components,
                       Number,
                       VectorizedArrayType>
            integrator(
              matrix_free, range, dof_no, quad_no, first_selected_component);

          // new
          const unsigned int dofs_per_component = integrator.dofs_per_component;
          const unsigned int dofs_per_cell      = integrator.dofs_per_cell;
          const unsigned int dofs_per_block =
            matrix_free.get_dof_handler(dof_no).n_dofs();

          std::vector<types::global_dof_index> dof_indices(dofs_per_component);
          std::vector<types::global_dof_index> dof_indices_mf(
            dofs_per_component);
          std::vector<types::global_dof_index> dof_indices_mf_all(
            dofs_per_cell);

          std::array<FullMatrix<typename MatrixType::value_type>,
                     VectorizedArrayType::size()>
            matrices;

          std::fill_n(matrices.begin(),
                      VectorizedArrayType::size(),
                      FullMatrix<typename MatrixType::value_type>(
                        dofs_per_cell, dofs_per_cell));

          const auto lexicographic_numbering =
            matrix_free
              .get_shape_info(dof_no,
                              quad_no,
                              first_selected_component,
                              integrator.get_active_fe_index(),
                              integrator.get_active_quadrature_index())
              .lexicographic_numbering;

          for (auto cell = range.first; cell < range.second; ++cell)
            {
              integrator.reinit(cell);

              const unsigned int n_filled_lanes =
                matrix_free.n_active_entries_per_cell_batch(cell);

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                matrices[v] = 0.0;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    integrator.begin_dof_values()[i] =
                      static_cast<Number>(i == j);

                  local_vmult(integrator);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int v = 0; v < n_filled_lanes; ++v)
                      matrices[v](i, j) = integrator.begin_dof_values()[i][v];
                }

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                {
                  const auto cell_v =
                    matrix_free.get_cell_iterator(cell, v, dof_no);

                  if (matrix_free.get_mg_level() !=
                      numbers::invalid_unsigned_int)
                    cell_v->get_mg_dof_indices(dof_indices);
                  else
                    cell_v->get_dof_indices(dof_indices);

                  for (unsigned int j = 0; j < dof_indices.size(); ++j)
                    dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];

                  // new
                  for (unsigned int b = 0, c = 0; b < n_components; ++b)
                    for (unsigned int i = 0; i < dofs_per_component; ++i, ++c)
                      if (true)
                        dof_indices_mf_all[c] =
                          dof_indices_mf[i] * n_components + b;
                      else
                        dof_indices_mf_all[c] =
                          dof_indices_mf[i] + b * dofs_per_block;

                  // new: remove small entries (TODO: only for FE_Q_iso_1)
                  Number max = 0.0;

                  for (unsigned int i = 0; i < matrices[v].m(); ++i)
                    for (unsigned int j = 0; j < matrices[v].n(); ++j)
                      max = std::max(max, std::abs(matrices[v][i][j]));

                  for (unsigned int i = 0; i < matrices[v].m(); ++i)
                    for (unsigned int j = 0; j < matrices[v].n(); ++j)
                      if (std::abs(matrices[v][i][j]) < 1e-10 * max)
                        matrices[v][i][j] = 0.0;

                  constraints.distribute_local_to_global(matrices[v],
                                                         dof_indices_mf_all,
                                                         dst);
                }
            }
        },
        matrix,
        matrix);

      matrix.compress(VectorOperation::add);
    }

    template <typename CLASS,
              int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      TrilinosWrappers::SparseMatrix &                    matrix,
      void (CLASS::*cell_operation)(FEEvaluation<dim,
                                                 fe_degree,
                                                 n_q_points_1d,
                                                 n_components,
                                                 Number,
                                                 VectorizedArrayType> &) const,
      const CLASS *      owning_class,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0)
    {
      compute_matrix<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType>(
        matrix_free,
        constraints,
        matrix,
        [&](auto &feeval) { (owning_class->*cell_operation)(feeval); },
        dof_no,
        quad_no,
        first_selected_component);
    }
    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints_in,
      std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &matrix,
      const std::function<void(FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components,
                                            Number,
                                            VectorizedArrayType> &)>
        &                local_vmult,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0)
    {
      using MatrixType = TrilinosWrappers::SparseMatrix;

      // new
      AssertDimension(
        matrix_free.get_dof_handler(dof_no).get_fe().n_components(), 1);

      std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
        constraints_for_matrix;
      const AffineConstraints<typename MatrixType::value_type> &constraints =
        dealii::MatrixFreeTools::internal::
          create_new_affine_constraints_if_needed(*matrix[0],
                                                  constraints_in,
                                                  constraints_for_matrix);

      double dummy;

      matrix_free.template cell_loop<double, double>(
        [&](const auto &, auto &, const auto &, const auto range) {
          FEEvaluation<dim,
                       fe_degree,
                       n_q_points_1d,
                       n_components,
                       Number,
                       VectorizedArrayType>
            integrator(
              matrix_free, range, dof_no, quad_no, first_selected_component);

          // new
          const unsigned int dofs_per_component = integrator.dofs_per_component;
          const unsigned int dofs_per_cell      = integrator.dofs_per_cell;
          // const unsigned int dofs_per_block =
          //  matrix_free.get_dof_handler(dof_no).n_dofs();

          std::vector<types::global_dof_index> dof_indices(dofs_per_component);
          std::vector<types::global_dof_index> dof_indices_mf(
            dofs_per_component);
          std::vector<types::global_dof_index> dof_indices_mf_all(
            dofs_per_cell);

          std::array<FullMatrix<typename MatrixType::value_type>,
                     VectorizedArrayType::size()>
            matrices;

          std::fill_n(matrices.begin(),
                      VectorizedArrayType::size(),
                      FullMatrix<typename MatrixType::value_type>(
                        dofs_per_cell, dofs_per_cell));

          const auto lexicographic_numbering =
            matrix_free
              .get_shape_info(dof_no,
                              quad_no,
                              first_selected_component,
                              integrator.get_active_fe_index(),
                              integrator.get_active_quadrature_index())
              .lexicographic_numbering;

          for (auto cell = range.first; cell < range.second; ++cell)
            {
              integrator.reinit(cell);

              const unsigned int n_filled_lanes =
                matrix_free.n_active_entries_per_cell_batch(cell);

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                matrices[v] = 0.0;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    integrator.begin_dof_values()[i] =
                      static_cast<Number>(i == j);

                  local_vmult(integrator);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int v = 0; v < n_filled_lanes; ++v)
                      matrices[v](i, j) = integrator.begin_dof_values()[i][v];
                }

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                {
                  const auto cell_v =
                    matrix_free.get_cell_iterator(cell, v, dof_no);

                  if (matrix_free.get_mg_level() !=
                      numbers::invalid_unsigned_int)
                    cell_v->get_mg_dof_indices(dof_indices);
                  else
                    cell_v->get_dof_indices(dof_indices);

                  for (unsigned int j = 0; j < dof_indices.size(); ++j)
                    dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];

                  // new
                  for (unsigned int b = 0; b < n_components; ++b)
                    {
                      FullMatrix<typename MatrixType::value_type> block(
                        dofs_per_component, dofs_per_component);

                      for (unsigned int i = 0; i < dofs_per_component; ++i)
                        for (unsigned int j = 0; j < dofs_per_component; ++j)
                          {
                            block(i, j) =
                              matrices[v](b * dofs_per_component + i,
                                          b * dofs_per_component + j);
                          }

                      constraints.distribute_local_to_global(block,
                                                             dof_indices_mf,
                                                             *matrix[b]);
                    }
                }
            }
        },
        dummy,
        dummy);

      for (unsigned int b = 0; b < matrix.size(); ++b)
        matrix[b]->compress(VectorOperation::add);
    }

    template <typename CLASS,
              int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const AffineConstraints<Number> &                             constraints,
      std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &matrix,
      void (CLASS::*cell_operation)(FEEvaluation<dim,
                                                 fe_degree,
                                                 n_q_points_1d,
                                                 n_components,
                                                 Number,
                                                 VectorizedArrayType> &) const,
      const CLASS *      owning_class,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0)
    {
      compute_matrix<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType>(
        matrix_free,
        constraints,
        matrix,
        [&](auto &feeval) { (owning_class->*cell_operation)(feeval); },
        dof_no,
        quad_no,
        first_selected_component);
    }

    /* This is a special wrapper to overcome a bug in Intel Compiler existing at
     * least in version 2021.2.0. For some reason, compiler can not choose
     * between the 2 native deal.II versions of MatrixFree::cell_loop() for
     * const and non-const object pointers if the pointer is 'this'. Inside the
     * wrapper, when the initially provided 'this' pointer has a distinct name,
     * the problem disappears.
     *
     * This bug also appears for class OperatorBase while for class Sintering
     * there is no conflict, that's why the wrapper is not used there. It seems,
     * the bug has something to do with inheritance.
     */
    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename CLASS,
              typename OutVector,
              typename InVector>
    void
    cell_loop_wrapper(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      void (CLASS::*cell_operation)(
        const MatrixFree<dim, Number, VectorizedArrayType> &,
        OutVector &,
        const InVector &,
        const std::pair<unsigned int, unsigned int> &) const,
      const CLASS *   owning_class,
      OutVector &     dst,
      const InVector &src,
      const bool      zero_dst_vector = false)
    {
      matrix_free.cell_loop(
        cell_operation, owning_class, dst, src, zero_dst_vector);
    }

    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename CLASS,
              typename OutVector,
              typename InVector>
    void
    cell_loop_wrapper(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      void (CLASS::*cell_operation)(
        const MatrixFree<dim, Number, VectorizedArrayType> &,
        OutVector &,
        const InVector &,
        const std::pair<unsigned int, unsigned int> &) const,
      const CLASS *   owning_class,
      OutVector &     dst,
      const InVector &src,
      const std::function<void(const unsigned int, const unsigned int)>
        &operation_before_loop,
      const std::function<void(const unsigned int, const unsigned int)>
        &                operation_after_loop,
      const unsigned int dof_handler_index_pre_post = 0)
    {
      matrix_free.cell_loop(cell_operation,
                            owning_class,
                            dst,
                            src,
                            operation_before_loop,
                            operation_after_loop,
                            dof_handler_index_pre_post);
    }

  } // namespace MyMatrixFreeTools
} // namespace dealii
