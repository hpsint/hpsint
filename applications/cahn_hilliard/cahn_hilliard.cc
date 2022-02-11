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


// Cahn-Hilliard equation with one phase.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

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
  Point<dim>   point{50.0, 50.0};
  const double rad = 20.0;

public:
  InitialValues()
    : Function<dim>(2)
  {}

  virtual double
  value(const dealii::Point<dim> &p,
        const unsigned int        component = 0) const override
  {
    if (component == 0)
      {
        double center[12][3] = {{0.1, 0.3, 0},
                                {0.8, 0.7, 0},
                                {0.5, 0.2, 0},
                                {0.4, 0.4, 0},
                                {0.3, 0.9, 0},
                                {0.8, 0.1, 0},
                                {0.9, 0.5, 0},
                                {0.0, 0.1, 0},
                                {0.1, 0.6, 0},
                                {0.5, 0.6, 0},
                                {1, 1, 0},
                                {0.7, 0.95, 0}};
        double rad[12]       = {12, 14, 19, 16, 11, 12, 17, 15, 20, 10, 11, 14};
        double dist;
        double scalar_IC = 0;
        for (unsigned int i = 0; i < 12; i++)
          {
            dist = 0.0;
            for (unsigned int dir = 0; dir < dim; dir++)
              {
                dist += (p[dir] - center[i][dir] * 100 /*TODO*/) *
                        (p[dir] - center[i][dir] * 100 /*TODO*/);
              }
            dist = std::sqrt(dist);

            scalar_IC += 0.5 * (1.0 - std::tanh((dist - rad[i]) / 1.5));
          }
        if (scalar_IC > 1.0)
          scalar_IC = 1.0;

        return scalar_IC;
      }
    else
      {
        return 0.0;
      }

    // if(component==1)
    //  return 0.0;
    //
    // double dist = point.distance(p);
    // return 0.5 * (1.0 - std::tanh(2 * (dist - rad)));
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

  MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
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
      phi(matrix_free);

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, true, false, false);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(phi.get_value(q), q);
            phi.integrate_scatter(true, false, dst);
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
};



template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  void
  run()
  {
    // geometry
    const double size = 100.0;

    // mesh
    const unsigned int n_refinements  = 6;
    const unsigned int n_subdivisions = 1;

    // time discretization
    const unsigned int n_time_steps        = 100000;
    const unsigned int n_time_steps_output = 1000;
    const double       dt                  = 0.001;

    //  model constants
    const double M     = 1.0;
    const double kappa = 1.5;

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, 0, size);
    tria.refine_global(n_refinements);

    FESystem<dim>   fe(FE_Q<dim>{fe_degree}, 2);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ<dim> mapping(1);

    QGaussLobatto<1> quad(n_points_1D);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

    VectorType src, dst;

    matrix_free.initialize_dof_vector(src);
    matrix_free.initialize_dof_vector(dst);

    VectorTools::interpolate(mapping, dof_handler, InitialValues<dim>(), src);

    const auto df_dc = [&](const auto &c) {
      return 4.0 * c * (c - 1.0) * (c - 0.5);
    };

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(src, "solution");
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;


      std::cout << "outputing at " << t << std::endl;

      std::ofstream output("solution." + std::to_string(counter++) + ".vtk");
      data_out.write_vtk(output);
    };

    FEEvaluation<dim, fe_degree, n_points_1D, 2, Number, VectorizedArrayType>
      phi(matrix_free);


    output_result(0.0);

    // time loop
    unsigned int counter = 0;
    for (double t = 0; counter++ < n_time_steps; t += dt)
      {
        // compute right-hand side vector
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, auto cells) {
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                phi.reinit(cell);
                phi.gather_evaluate(src, true, true, false);
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  {
                    const auto value    = phi.get_value(q);
                    const auto gradient = phi.get_gradient(q);

                    Tensor<1, 2, VectorizedArrayType> value_result;
                    value_result[0] = value[0];
                    value_result[1] = df_dc(value[0]);

                    Tensor<1, 2, Tensor<1, dim, VectorizedArrayType>>
                      gradient_result;
                    gradient_result[0] = -dt * M * gradient[1];
                    gradient_result[1] = kappa * gradient[0];

                    phi.submit_value(value_result, q);
                    phi.submit_gradient(gradient_result, q);
                  }
                phi.integrate_scatter(true, true, dst);
              }
          },
          dst,
          src,
          true);

        // invert mass matrix
        ReductionControl     reduction_control;
        SolverCG<VectorType> solver(reduction_control);
        solver.solve(MassMatrix<dim,
                                fe_degree,
                                n_points_1D,
                                2,
                                Number,
                                VectorizedArrayType>(matrix_free),
                     src,
                     dst,
                     PreconditionIdentity());
        std::cout << "it " << counter << ": " << reduction_control.last_step()
                  << std::endl;

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
  Test<2, 2>                       runner;
  runner.run();
}