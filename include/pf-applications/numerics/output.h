// ---------------------------------------------------------------------
//
// Copyright (C) 2024 - 2026 by the hpsint authors
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

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/numerics/data_out.h>

using namespace dealii;

namespace Sintering
{
  namespace Postprocessors
  {
    template <typename T>
    struct BlockVectorWrapper;

    template <typename T>
    struct BlockVectorWrapper<std::vector<T>>
    {
      using BlockType = T;

      BlockVectorWrapper(const std::vector<T> &v)
      {
        data.reserve(v.size());
        std::transform(v.begin(), v.end(), data.begin(), [](const auto &item) {
          return &item;
        });
      }

      template <typename Iterator>
      BlockVectorWrapper(Iterator begin, Iterator end)
      {
        data.reserve(std::distance(begin, end));
        std::transform(begin, end, data.begin(), [](const auto &item) {
          return &item;
        });
      }

      const typename std::vector<T>::value_type &
      block(typename std::vector<T>::size_type i) const
      {
        return *data[i];
      }

      typename std::vector<T>::size_type
      n_blocks() const
      {
        return data.size();
      }

    private:
      std::vector<const T *> data;
    };

    template <int dim, typename VectorType>
    DataOutWithRanges<dim>
    build_default_output(const DoFHandler<dim>          &dof_handler,
                         const VectorType               &solution,
                         const std::vector<std::string> &names,
                         const bool                      add_subdomains = true,
                         const bool higher_order_cells                  = false)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = higher_order_cells;

      DataOutWithRanges<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.set_flags(flags);

      for (unsigned int b = 0; b < solution.n_blocks(); ++b)
        if (b < names.size() && !names[b].empty())
          data_out.add_data_vector(solution.block(b), names[b]);

      // Output subdomain structure
      if (add_subdomains)
        {
          auto subdomain_id =
            dof_handler.get_triangulation().locally_owned_subdomain();
          if (subdomain_id == numbers::invalid_subdomain_id)
            subdomain_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

          Vector<float> subdomain(
            dof_handler.get_triangulation().n_active_cells());
          for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain[i] = subdomain_id;

          data_out.add_data_vector(subdomain, "subdomain");
        }

      return data_out;
    }

    // Consistent-mass L2 projection helpers (for derived output quantities).
    // Derived quantities are nonlinear, generally discontinuous functions of
    // the FE solution. Writing them for visualization requires projecting them
    // onto the continuous FE_Q space.
    //
    // Matrix-free consistent scalar FE_Q mass operator: dst = M * src
    // (quadrature index 0, single scalar component at dof_index).
    template <int dim,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    struct ScalarMassOperator
    {
      const MatrixFree<dim, Number, VectorizedArrayType> &mf;
      unsigned int                                        dof_index;

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        const unsigned int di = dof_index;

        const std::function<
          void(const MatrixFree<dim, Number, VectorizedArrayType> &,
               VectorType &,
               const VectorType &,
               const std::pair<unsigned int, unsigned int> &)>
          cell_op =
            [di](const MatrixFree<dim, Number, VectorizedArrayType> &data,
                 VectorType                                         &dst,
                 const VectorType                                   &src,
                 const std::pair<unsigned int, unsigned int>        &range) {
              FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> eval(
                data, di);

              for (auto cell = range.first; cell < range.second; ++cell)
                {
                  eval.reinit(cell);
                  eval.read_dof_values(src);
                  eval.evaluate(EvaluationFlags::values);
                  for (unsigned int q = 0; q < eval.n_q_points; ++q)
                    eval.submit_value(eval.get_value(q), q);
                  eval.integrate(EvaluationFlags::values);
                  eval.distribute_local_to_global(dst);
                }
            };

        mf.cell_loop(cell_op, dst, src, /*zero_dst_before_use=*/true);
      }
    };

    // Solve the consistent-mass L2 projection M x = b in place for each entry.
    // On entry each vector holds the assembled load vector b_i = (phi_i, q);
    // on exit it holds the projected field x with hanging-node constraints
    // distributed.
    template <int dim,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    void
    solve_l2_projection(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number>                    &constraints,
      const unsigned int                                  dof_index,
      std::vector<VectorType>                            &vectors)
    {
      ScalarMassOperator<dim, Number, VectorizedArrayType> mass{matrix_free,
                                                                dof_index};

      VectorType solution;
      matrix_free.initialize_dof_vector(solution, dof_index);

      for (auto &rhs : vectors)
        {
          rhs.compress(VectorOperation::add);

          ReductionControl     control(1000, 1e-20, 1e-8);
          SolverCG<VectorType> cg(control);

          solution = 0.0;
          cg.solve(mass, solution, rhs, PreconditionIdentity());
          constraints.distribute(solution);

          rhs = solution;
        }
    }
  } // namespace Postprocessors
} // namespace Sintering