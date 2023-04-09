// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

// Test performance of the sintering operator

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <pf-applications/base/performance.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

using namespace dealii;

//#define USE_SUBDIVISIONS


template <int dim, int n_components, typename Number, typename VectorType>
void
helmholtz_operator_fe_values_0(VectorType &           dst,
                               const VectorType &     src,
                               const Mapping<dim> &   mapping,
                               const DoFHandler<dim> &dof_handler,
                               const Quadrature<dim> &quadrature)
{
  const auto &fe = dof_handler.get_fe();

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  std::vector<Tensor<1, n_components, Number>> values(
    fe_values.n_quadrature_points);
  std::vector<Tensor<1, n_components, Tensor<1, dim, Number>>> gradients(
    fe_values.n_quadrature_points);

  Vector<Number> src_local(fe_values.dofs_per_cell);
  Vector<Number> dst_local(fe_values.dofs_per_cell);

  src.update_ghost_values();
  dst = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      cell->get_dof_values(src, src_local); // TODO: constraints

      for (const auto q : fe_values.quadrature_point_indices())
        for (unsigned int c = 0; c < n_components; ++c)
          {
            Number                 value = 0.0;
            Tensor<1, dim, Number> gradient;

            for (const auto i : fe_values.dof_indices())
              {
                value +=
                  src_local[i] * fe_values.shape_value_component(i, q, c);
                gradient +=
                  src_local[i] * fe_values.shape_grad_component(i, q, c);
              }

            values[q][c]    = value;
            gradients[q][c] = gradient;
          }

      for (const auto q : fe_values.quadrature_point_indices())
        {
          values[q] *= fe_values.JxW(q);
          gradients[q] *= fe_values.JxW(q);
        }

      for (const auto i : fe_values.dof_indices())
        {
          dst_local[i] = 0.0;

          for (const auto q : fe_values.quadrature_point_indices())
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  dst_local[i] +=
                    values[q][c] * fe_values.shape_value_component(i, q, c) +
                    gradients[q][c] * fe_values.shape_grad_component(i, q, c);
                }
            }
        }

      cell->distribute_local_to_global(dst_local, dst); // TODO: constraints
    }

  src.zero_out_ghost_values();
  dst.compress(VectorOperation::add);
}


template <int dim, int n_components, typename Number, typename VectorType>
void
helmholtz_operator_fe_values_1(VectorType &           dst,
                               const VectorType &     src,
                               const Mapping<dim> &   mapping,
                               const DoFHandler<dim> &dof_handler,
                               const Quadrature<dim> &quadrature)
{
  const auto &fe = dof_handler.get_fe();

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  std::vector<Tensor<1, n_components, Number>> values(
    fe_values.n_quadrature_points);
  std::vector<Tensor<1, n_components, Tensor<1, dim, Number>>> gradients(
    fe_values.n_quadrature_points);

  Vector<Number> src_local(fe_values.dofs_per_cell);
  Vector<Number> dst_local(fe_values.dofs_per_cell);

  const unsigned int n_dofs_per_cell_scalar =
    fe_values.dofs_per_cell / n_components;

  src.update_ghost_values();
  dst = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      cell->get_dof_values(src, src_local); // TODO: constraints

      for (const auto q : fe_values.quadrature_point_indices())
        for (unsigned int c = 0; c < n_components; ++c)
          {
            Number                 value = 0.0;
            Tensor<1, dim, Number> gradient;

            for (unsigned int j = 0; j < n_dofs_per_cell_scalar; ++j)
              {
                const unsigned int i = fe.component_to_system_index(c, j);

                value +=
                  src_local[i] * fe_values.shape_value_component(i, q, c);
                gradient +=
                  src_local[i] * fe_values.shape_grad_component(i, q, c);
              }

            values[q][c]    = value;
            gradients[q][c] = gradient;
          }

      for (const auto q : fe_values.quadrature_point_indices())
        {
          values[q] *= fe_values.JxW(q);
          gradients[q] *= fe_values.JxW(q);
        }

      for (const auto i : fe_values.dof_indices())
        dst_local[i] = 0.0;

      for (unsigned int j = 0; j < n_dofs_per_cell_scalar; ++j)
        {
          for (const auto q : fe_values.quadrature_point_indices())
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  const unsigned int i = fe.component_to_system_index(c, j);
                  dst_local[i] +=
                    values[q][c] * fe_values.shape_value_component(i, q, c) +
                    gradients[q][c] * fe_values.shape_grad_component(i, q, c);
                }
            }
        }

      cell->distribute_local_to_global(dst_local, dst); // TODO: constraints
    }

  src.zero_out_ghost_values();
  dst.compress(VectorOperation::add);
}


template <int dim,
          int fe_degree,
          int n_q_points,
          int n_components,
          typename Number,
          typename VectorType>
void
helmholtz_operator_fe_evaluation(VectorType &                   dst,
                                 const VectorType &             src,
                                 const MatrixFree<dim, Number> &matrix_free)
{
  matrix_free.template cell_loop<VectorType, VectorType>(
    [](const auto &data, auto &dst, const auto &src, const auto &range) {
      FEEvaluation<dim, fe_degree, n_q_points, n_components, Number> phi(data);

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);

          phi.gather_evaluate(src,
                              EvaluationFlags::values |
                                EvaluationFlags::gradients);

          for (unsigned int q_index = 0; q_index < phi.n_q_points; ++q_index)
            {
              phi.submit_value(phi.get_value(q_index), q_index);
              phi.submit_gradient(phi.get_gradient(q_index), q_index);
            }

          phi.integrate_scatter(EvaluationFlags::values |
                                  EvaluationFlags::gradients,
                                dst);
        }
    },
    dst,
    src,
    true);
}

#define EXPAND_OPERATIONS(OPERATION)             \
  switch (n_components)                          \
    {                                            \
      case 1:                                    \
        {                                        \
          OPERATION(1, 0);                       \
        }                                        \
        break;                                   \
      case 2:                                    \
        {                                        \
          OPERATION(2, 0);                       \
        }                                        \
        break;                                   \
      case 3:                                    \
        {                                        \
          OPERATION(3, 0);                       \
        }                                        \
        break;                                   \
      case 4:                                    \
        {                                        \
          OPERATION(4, 0);                       \
        }                                        \
        break;                                   \
      default:                                   \
        AssertThrow(false, ExcNotImplemented()); \
    }


// clang-format off
/**
 * likwid-mpirun -np 40 -f -g CACHES   -m -O ./applications/sintering/sintering-throughput
 * likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./applications/sintering/sintering-throughput
 */
// clang-format on
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  constexpr unsigned int dim = 3;
#ifdef USE_SUBDIVISIONS
  constexpr unsigned int fe_degree            = 2;
  constexpr unsigned int n_q_points           = fe_degree * 2;
  constexpr unsigned int n_global_refinements = 6; // TODO
  const std::string      fe_type              = "FE_Q_ISO_Q1";
#else
  constexpr unsigned int fe_degree            = 1;
  constexpr unsigned int n_q_points           = fe_degree + 1;
  constexpr unsigned int n_global_refinements = 7; // TODO
  const std::string      fe_type              = "FE_Q";
#endif
  constexpr unsigned int n_repetitions = 1;
  using Number                         = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  ConvergenceTable table;

  for (unsigned int n_components = 1; n_components <= 4; ++n_components)
    {
#ifdef USE_SUBDIVISIONS
      const FESystem<dim>  fe(FE_Q_iso_Q1<dim>(fe_degree), n_components);
      const QIterated<dim> quadrature(QGauss<1>(2), fe_degree);
#else
      const FESystem<dim> fe(FE_Q<dim>(fe_degree), n_components);
      const QGauss<dim>   quadrature(n_q_points);
#endif

      const MappingQ1<dim> mapping;

      parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
      GridGenerator::hyper_cube(tria);
      tria.refine_global(n_global_refinements);

      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      AffineConstraints<Number> constraints;

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags =
        update_values | update_gradients | update_quadrature_points;
      additional_data.overlap_communication_computation = false;

      MatrixFree<dim, Number> matrix_free;
      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);

      table.add_value("dim", dim);
      table.add_value("fe_type", fe_type);
      table.add_value("n_dofs", dof_handler.n_dofs());
      table.add_value("n_components", n_components);

      VectorType src, dst;
      matrix_free.initialize_dof_vector(src);
      matrix_free.initialize_dof_vector(dst);

      src = 1.0;

#define OPERATION(c, d)                                                        \
  const auto time_0 = run_measurement(                                         \
    [&]() {                                                                    \
      helmholtz_operator_fe_values_0<dim, c, Number>(                          \
        dst, src, mapping, dof_handler, quadrature);                           \
    },                                                                         \
    n_repetitions);                                                            \
  table.add_value("t_0", time_0);                                              \
  table.set_scientific("t_0", true);                                           \
  table.add_value("l_0", dst.l2_norm());                                       \
  table.set_scientific("l_0", true);                                           \
                                                                               \
  const auto time_1 = run_measurement(                                         \
    [&]() {                                                                    \
      helmholtz_operator_fe_values_1<dim, c, Number>(                          \
        dst, src, mapping, dof_handler, quadrature);                           \
    },                                                                         \
    n_repetitions);                                                            \
  table.add_value("t_1", time_1);                                              \
  table.set_scientific("t_1", true);                                           \
  table.add_value("l_1", dst.l2_norm());                                       \
  table.set_scientific("l_1", true);                                           \
                                                                               \
  const auto time_2 = run_measurement(                                         \
    [&]() {                                                                    \
      helmholtz_operator_fe_evaluation<dim, fe_degree, n_q_points, c, Number>( \
        dst, src, matrix_free);                                                \
    },                                                                         \
    n_repetitions);                                                            \
  table.add_value("t_2", time_2);                                              \
  table.set_scientific("t_2", true);                                           \
  table.add_value("l_2", dst.l2_norm());                                       \
  table.set_scientific("l_2", true);


      EXPAND_OPERATIONS(OPERATION);

#undef OPERATION
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
