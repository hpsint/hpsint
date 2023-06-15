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


// Allen-Cahn equation with two phase (using FESystem).

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;

template <int dim>
class InitialValues : public dealii::Function<dim>
{
private:
  Point<dim>   p_1{25.0, 50.0};
  Point<dim>   p_2{75.0, 50.0};
  const double rad_1 = 20.0;
  const double rad_2 = 15.0;

  const unsigned int phase;

public:
  InitialValues(const unsigned int phase)
    : Function<dim>(1)
    , phase(phase)
  {}

  virtual double
  value(const dealii::Point<dim> &p,
        const unsigned int        component = 0) const override
  {
    AssertDimension(component, 0);
    (void)component;

    if (phase == 0)
      return 0.5 * (1.0 - std::tanh(2 * (p_1.distance(p) - rad_1)));
    else
      return 0.5 * (1.0 - std::tanh(2 * (p_2.distance(p) - rad_2)));
  }
};



template <int dim,
          int degree,
          int n_points_1D,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
class MassMatrix
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const unsigned int                                  dof_index)
    : matrix_free(matrix_free)
    , dof_index(dof_index)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi(matrix_free, dof_index);

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(phi.get_value(q), q);
            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      true);
  }

  void
  initialize_dof_vector(VectorType &dst) const
  {
    matrix_free.initialize_dof_vector(dst);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const unsigned int                                  dof_index;
};



template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::BlockVector<Number>;

  void
  run()
  {
    // geometry
    const double size = 100.0;

    // mesh
    const unsigned int n_refinements  = 7;
    const unsigned int n_subdivisions = 1;

    // time discretization
    const unsigned int n_time_steps        = 1000;
    const unsigned int n_time_steps_output = 20;
    const double       dt                  = 0.01;

    //  model constants
    const double M     = 1.0;
    const double kappa = 0.5;

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, 0, size);
    tria.refine_global(n_refinements);

    FE_Q<dim>       fe_1(fe_degree);
    DoFHandler<dim> dof_handler_1(tria);
    dof_handler_1.distribute_dofs(fe_1);

    FE_Q<dim>       fe_2(fe_degree);
    DoFHandler<dim> dof_handler_2(tria);
    dof_handler_2.distribute_dofs(fe_2);

    MappingQ<dim> mapping(1);

    QGauss<1> quad(n_points_1D);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(
      mapping,
      std::vector<const DoFHandler<dim> *>{&dof_handler_1, &dof_handler_2},
      std::vector<const AffineConstraints<Number> *>{&constraint, &constraint},
      std::vector<Quadrature<1>>{quad},
      additional_data);

    VectorType src(2), dst(2);

    matrix_free.initialize_dof_vector(src.block(0), 0);
    matrix_free.initialize_dof_vector(dst.block(0), 0);
    matrix_free.initialize_dof_vector(src.block(1), 1);
    matrix_free.initialize_dof_vector(dst.block(1), 1);

    VectorTools::interpolate(mapping,
                             dof_handler_1,
                             InitialValues<dim>(0),
                             src.block(0));
    VectorTools::interpolate(mapping,
                             dof_handler_2,
                             InitialValues<dim>(1),
                             src.block(1));

    const auto df_dphi = [&](const auto &phi) {
      return phi * phi * phi * 2.0 - phi * 2.0;
    };

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.add_data_vector(dof_handler_1, src.block(0), "solution_0");
      data_out.add_data_vector(dof_handler_2, src.block(1), "solution_1");
      data_out.build_patches(mapping, fe_degree);

      std::cout << "outputing at " << t << std::endl;

      static unsigned int counter = 0;
      std::ofstream output("solution." + std::to_string(counter++) + ".vtk");
      data_out.write_vtk(output);
    };

    FEEvaluation<dim, fe_degree, n_points_1D, 1, Number, VectorizedArrayType>
      phi_1(matrix_free, 0);
    FEEvaluation<dim, fe_degree, n_points_1D, 1, Number, VectorizedArrayType>
      phi_2(matrix_free, 1);


    output_result(0.0);


    // time loop
    unsigned int counter = 0;
    for (double t = 0; counter++ < n_time_steps; t += dt)
      {
        // compute right-hand side vector
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, auto cells) {
            (void)dst;
            (void)src;

            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi_1.reinit(cell);
                phi_1.gather_evaluate(src.block(0),
                                      EvaluationFlags::values |
                                        EvaluationFlags::gradients);

                phi_2.reinit(cell);
                phi_2.gather_evaluate(src.block(1),
                                      EvaluationFlags::values |
                                        EvaluationFlags::gradients);

                for (unsigned int q = 0; q < phi_1.n_q_points; ++q)
                  {
                    phi_1.submit_gradient(-dt * M * kappa *
                                            phi_1.get_gradient(q),
                                          q);
                    phi_2.submit_gradient(-dt * M * kappa *
                                            phi_2.get_gradient(q),
                                          q);

                    const auto val_1 = phi_1.get_value(q);
                    phi_1.submit_value(val_1 - dt * M * df_dphi(val_1), q);
                    const auto val_2 = phi_2.get_value(q);
                    phi_2.submit_value(val_2 - dt * M * df_dphi(val_2), q);
                  }

                phi_1.integrate_scatter(EvaluationFlags::values |
                                          EvaluationFlags::gradients,
                                        dst.block(0));
                phi_2.integrate_scatter(EvaluationFlags::values |
                                          EvaluationFlags::gradients,
                                        dst.block(1));
              }
          },
          dst,
          src,
          true);

        // invert mass matrix
        {
          ReductionControl reduction_control;
          SolverCG<LinearAlgebra::distributed::Vector<Number>> solver(
            reduction_control);
          solver.solve(MassMatrix<dim,
                                  fe_degree,
                                  n_points_1D,
                                  1,
                                  Number,
                                  VectorizedArrayType>(matrix_free, 0),
                       src.block(0),
                       dst.block(0),
                       PreconditionIdentity());
          std::cout << "it-1 " << counter << ": "
                    << reduction_control.last_step() << std::endl;
        }


        {
          ReductionControl reduction_control;
          SolverCG<LinearAlgebra::distributed::Vector<Number>> solver(
            reduction_control);
          solver.solve(MassMatrix<dim,
                                  fe_degree,
                                  n_points_1D,
                                  1,
                                  Number,
                                  VectorizedArrayType>(matrix_free, 1),
                       src.block(1),
                       dst.block(1),
                       PreconditionIdentity());
          std::cout << "it-2 " << counter << ": "
                    << reduction_control.last_step() << std::endl;
        }

        if (counter % n_time_steps_output == 0)
          output_result(t);
      }
  }

private:
};


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Test<2, 1>                       runner;
  runner.run();
}