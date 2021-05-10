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

// Allen-Cahn equation with one phase.


#include <deal.II/base/conditional_ostream.h>
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

#include <deal.II/sundials/kinsol.h>

#include "include/newton.h"

//#define IDENTITY

using namespace dealii;

template <int dim>
class InitialValues : public dealii::Function<dim>
{
private:
  Point<dim>   point{50.0, 50.0};
  const double rad = 20.0;

public:
  InitialValues()
    : Function<dim>(1)
  {}

  virtual double
  value(const dealii::Point<dim> &p,
        const unsigned int        component = 0) const override
  {
    (void)component;
    double dist = point.distance(p);
    return 0.5 * (1.0 - std::tanh(2 * (dist - rad)));
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
  using VectorType = Vector<Number>;

  using value_type  = Number;
  using vector_type = VectorType;

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
          int degree,
          int n_points_1D,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
class AllenCahnImplicit
{
public:
  using VectorType = Vector<Number>;

  using value_type  = Number;
  using vector_type = VectorType;

  AllenCahnImplicit(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const double                                        dt,
    const double                                        M,
    const double                                        kappa)
    : matrix_free(matrix_free)
    , dt(dt)
    , M(M)
    , kappa(kappa)
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

    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi_lin(matrix_free);

    // second derivative of potential with respect to phi
    const auto d2f_dphi2 = [&](const auto &phi) {
      return phi * phi * 6.0 - 2.0;
    };

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);
            phi_lin.reinit(cell);

            phi.gather_evaluate(src,
                                /*evaluate values = */ true,
                                /*evaluate gradients = */ true,
                                false);

            phi_lin.read_dof_values_plain(src);
            phi_lin.evaluate(true, false, false);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto val = phi_lin.get_value(q);
                phi.submit_value((1.0 + dt * M * d2f_dphi2(val)) *
                                   phi.get_value(q),
                                 q);

                phi.submit_gradient(+dt * M * kappa * phi.get_gradient(q), q);
              }
            phi.integrate_scatter(/*evaluate values = */ true,
                                  /*evaluate gradients = */ true,
                                  dst);
          }
      },
      dst,
      src,
      true);
  }

  void
  evaluate_nonlinear_residual(VectorType &dst, const VectorType &src) const
  {
    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi_old(matrix_free);

    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi(matrix_free);

    // first derivative of potential with respect to phi
    const auto df_dphi = [&](const auto &phi) {
      return phi * phi * phi * 2.0 - phi * 2.0;
    };

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi_old.reinit(cell);
            phi.reinit(cell);

            phi.gather_evaluate(src,
                                /*evaluate values = */ true,
                                /*evaluate gradients = */ true,
                                false);

            // get values from old solution
            phi_old.read_dof_values_plain(old_solution);
            phi_old.evaluate(true, false, false);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                const auto val = phi.get_value(q);
                // get old value
                const auto old_val = phi_old.get_value(q);
                // value terms - is that really correct???
                phi.submit_value((1.0 + dt * M * df_dphi(val)) *
                                     phi.get_value(q) -
                                   old_val,
                                 q);
                // gradient terms
                phi.submit_gradient(+dt * M * kappa * phi.get_gradient(q), q);
              }
            phi.integrate_scatter(/*evaluate values = */ true,
                                  /*evaluate gradients = */ true,
                                  dst);
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

  void
  set_solution_linearization(const VectorType &src) const
  {
    this->solution_linearization = src;
  }

  void
  set_previous_solution(const VectorType &src) const
  {
    this->old_solution = src;
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const double                                        dt;
  const double                                        M;
  const double                                        kappa;

  mutable VectorType old_solution;
  mutable VectorType solution_linearization;
};



template <typename Operator>
class InverseMassMatrix
{
public:
  using VectorType = typename Operator::vector_type;

  InverseMassMatrix(const Operator &op)
    : op(op)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    // invert mass matrix
    ReductionControl     reduction_control;
    SolverCG<VectorType> solver(reduction_control);
    solver.solve(op, dst, src, PreconditionIdentity());
  }

  const Operator &op;
};


template <typename Operator, typename Preconditioner>
class SolverCGWrapper
{
public:
  using VectorType = typename Operator::vector_type;

  SolverCGWrapper(const Operator &op, const Preconditioner &preconditioner)
    : op(op)
    , preconditioner(preconditioner)
  {}

  unsigned int
  solve(VectorType &dst, const VectorType &src, const bool do_update)
  {
    (void)do_update; // no preconditioner is used

    ReductionControl     reduction_control;
    SolverCG<VectorType> solver(reduction_control);
    solver.solve(op, dst, src, preconditioner);

    return reduction_control.last_step();
  }

  const Operator &      op;
  const Preconditioner &preconditioner;
};


template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 3,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = Vector<Number>;

  void
  run()
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

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

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ<dim> mapping(1);

    QGauss<1> quad(n_points_1D);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

    VectorType solution;

    matrix_free.initialize_dof_vector(solution);

    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim>(),
                             solution);

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");

      solution.update_ghost_values();
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;

      pcout << "outputing at " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };

    FEEvaluation<dim, fe_degree, n_points_1D, 1, Number, VectorizedArrayType>
      phi(matrix_free);


    output_result(0.0);


    using Operator = AllenCahnImplicit<dim,
                                       fe_degree,
                                       n_points_1D,
                                       1,
                                       Number,
                                       VectorizedArrayType>;

#ifdef IDENTITY
    using Preconditioner = PreconditionIdentity;
#else
    using PreconditionerOperator =
      MassMatrix<dim, fe_degree, n_points_1D, 1, Number, VectorizedArrayType>;
    using Preconditioner = InverseMassMatrix<PreconditionerOperator>;
#endif
    using LinearSolver = SolverCGWrapper<Operator, Preconditioner>;

    Operator nonlinear_operator(matrix_free, dt, M, kappa);

#ifdef IDENTITY
    Preconditioner preconditioner;
#else
    PreconditionerOperator precondition_operator(matrix_free);
    Preconditioner         preconditioner(precondition_operator);
#endif

    LinearSolver linear_solver(nonlinear_operator, preconditioner);

#if false
    NewtonSolver<VectorType, Operator, LinearSolver> newton_solver(
      nonlinear_operator, linear_solver);
#endif

    // time loop
    unsigned int counter = 0;
    for (double t = 0; counter++ < n_time_steps; t += dt)
      {
        nonlinear_operator.set_previous_solution(solution);
#if false
        const auto statistics = newton_solver.solve(solution);

        pcout << "Solved in " << statistics.newton_iterations
              << " Newton iterations and " << statistics.linear_iterations
              << " linear iterations" << std::endl;

#else
        typename SUNDIALS::KINSOL<VectorType>::AdditionalData additional_data;
        additional_data.function_tolerance = 1e-8 /*TODO*/;

        // create non-liner solver and attach functions for ...
        SUNDIALS::KINSOL<VectorType> nonlinear_solver(additional_data);

        // ... initialize vector
        nonlinear_solver.reinit_vector = [&](VectorType &x) {
          nonlinear_operator.initialize_dof_vector(x);
        };

        // ... evaluate residual
        nonlinear_solver.residual = [&](const VectorType &evaluation_point,
                                        VectorType &      residual) {
          nonlinear_operator.evaluate_nonlinear_residual(residual,
                                                         evaluation_point);
          return 0;
        };

        // ... setup Jacobian (MatrixFree) simply set linearization point
        nonlinear_solver.setup_jacobian =
          [&](const VectorType &current_u, const VectorType & /*current_f*/) {
            nonlinear_operator.set_solution_linearization(current_u);
            return 0;
          };

        // ... solve linear system of equations
        nonlinear_solver.solve_with_jacobian =
          [&](const VectorType &rhs, VectorType &dst, const double tolerance) {
            (void)tolerance /*TODO*/;
            linear_solver.solve(dst, rhs, true /*TODO*/);
            return 0;
          };

        // ... update preconditioner (TODO)

        // solve!
        nonlinear_solver.solve(solution);
#endif

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