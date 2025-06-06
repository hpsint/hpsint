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


// Cahn-Hilliard equation with one phase.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
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

#include <pf-applications/lac/dynamic_block_vector.h>

#include <fstream>

using namespace dealii;

template <int dim, int component>
class InitialValues : public Function<dim>
{};

template <int dim>
class InitialValues<dim, 0> : public Function<dim>
{
private:
  Point<dim> center{0.5, 0.5};

  std::function<double(double)> phifunc;

public:
  InitialValues(double W)
    : Function<dim>(1)
  {
    phifunc = [=](double x) { return 0.5 * (1.0 + std::tanh(x / W)); };
  }

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    const auto dist = center.distance(p);
    return phifunc(0.25 - dist);
  }
};

template <int dim>
class InitialValues<dim, 1> : public Function<dim>
{
private:
  InitialValues<dim, 0> phi0;

public:
  InitialValues(double W)
    : Function<dim>(1)
    , phi0(W)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    return 1 - phi0.value(p, 0);
  }
};

template <int dim>
class InitialValues<dim, 2> : public Function<dim>
{
private:
  InitialValues<dim, 0> phi0;
  InitialValues<dim, 1> phi1;

  std::array<double, 2> c_0;

public:
  InitialValues(double W, std::array<double, 2> c_0)
    : Function<dim>(1)
    , phi0(W)
    , phi1(W)
    , c_0{c_0}
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    // Smooth interface
    // return phi0.value(p, 0) * c_0[0] + phi1.value(p, 0) * c_0[1];

    // Sharp interface for c
    return (phi0.value(p, 0) > phi1.value(p, 0)) ? c_0[0] : c_0[1];
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
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

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
};



template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  void
  run()
  {
    constexpr unsigned int n_comp = 3;
    constexpr unsigned int n_phi  = n_comp - 1;

    // geometry
    const double size = 1.0;

    // mesh
    const unsigned int n_refinements  = 6; // -> 64x64 grid
    const unsigned int n_subdivisions = 1;

    // time discretization
    const unsigned int n_time_steps        = 1000;
    const unsigned int n_time_steps_output = 100;

    const auto getW_if = [](const Number Wfac, const Number dx) {
      return Wfac * dx / std::log(99.0);
    };

    // Physical parameters
    const Number dx    = size / std::pow(2, n_refinements); // grid spacing
    const Number gamma = 1;
    const Number M     = 1;
    const Number dtfac = 0.98;
    const Number W     = getW_if(6, dx);
    const Number A     = 3 * W * gamma / 2;
    const Number B     = 6 * gamma / W;
    const Number L     = 2 * M / (3 * W);
    const Number D     = 1;

    std::cout << "dx: " << dx << std::endl;
    std::cout << "W: " << W << std::endl;
    std::cout << "A: " << A << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "D: " << D << std::endl;

    // Key expressions
    const std::array<Number, n_phi> k    = {{500.0, 500.0}};
    const std::array<Number, n_phi> c_0  = {{0.02, 0.98}};
    const unsigned int              vidx = 0;
    const unsigned int              sidx = 1;
    const std::array<Number, n_phi> linf = {{1.0, -1.0}};

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, 0, size);
    tria.refine_global(n_refinements);

    const auto create_fe = []() {
      if constexpr (fe_degree == 0)
        return FE_DGQ<dim>(fe_degree);
      else
        return FE_Q<dim>(fe_degree);
    };

    auto            fe = create_fe();
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

    VectorType src(n_comp), dst(n_comp), incr(n_comp);

    for (unsigned int c = 0; c < n_comp; ++c)
      {
        matrix_free.initialize_dof_vector(src.block(c));
        matrix_free.initialize_dof_vector(dst.block(c));
        matrix_free.initialize_dof_vector(incr.block(c));
      }

    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 0>(W),
                             src.block(0));
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 1>(W),
                             src.block(1));
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 2>(W, c_0),
                             src.block(2));

    const auto output_result = [&](const VectorType &vec, const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(vec.block(0),
                               "phi0",
                               DataOut_DoFData<dim, dim>::type_dof_data);
      data_out.add_data_vector(vec.block(1),
                               "phi1",
                               DataOut_DoFData<dim, dim>::type_dof_data);
      data_out.add_data_vector(vec.block(2),
                               "c",
                               DataOut_DoFData<dim, dim>::type_dof_data);
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;


      std::cout << "outputing at " << t << std::endl;

      std::ofstream output("solution." + std::to_string(counter++) + ".vtk");
      data_out.write_vtk(output);
    };

    FEEvaluation<dim,
                 fe_degree,
                 n_points_1D,
                 n_comp,
                 Number,
                 VectorizedArrayType>
      eval(matrix_free);

    MassMatrix<dim, fe_degree, n_points_1D, n_comp, Number, VectorizedArrayType>
      mass_matrix(matrix_free);

    output_result(src, 0.0);

    // Timestep estimates
    const Number dt_phi =
      3 * W * W * dx * dx * gamma /
      (M * (2 * 12 * W * W * gamma * 2 + 0.96 * dx * dx * gamma * k[0] +
            12 * dx * dx * gamma * gamma));

    const Number dt_c = dx * dx / (2 * 4 * D * (1 + std::abs(c_0[0] - c_0[1])));
    const Number dt   = dtfac * std::min(dt_c, dt_phi);

    // Some helper lambdas for the calculations
    const auto calc_ca = [&](const VectorizedArrayType                    &c,
                             const std::array<VectorizedArrayType, n_phi> &phi,
                             const unsigned int                            a) {
      const auto idx    = static_cast<unsigned>(a == 0);
      const auto prefac = linf[a];

      const auto nom = (c * k[idx] + prefac * (c_0[vidx] * k[vidx] * phi[idx] -
                                               c_0[sidx] * k[sidx] * phi[idx]));

      const auto denom = (k[0] * phi[1] + k[1] * phi[0]);

      return nom / denom;
    };

    const auto calc_grad_ca =
      [&](
        const VectorizedArrayType                                    &c,
        const Tensor<1, dim, VectorizedArrayType>                    &c_grad,
        const std::array<VectorizedArrayType, n_phi>                 &phi,
        const std::array<Tensor<1, dim, VectorizedArrayType>, n_phi> &phi_grad,
        const unsigned int                                            a) {
        const auto idx    = static_cast<unsigned>(a == 0);
        const auto prefac = linf[a];

        const auto coeff = prefac * (c_0[vidx] * k[vidx] - c_0[sidx] * k[sidx]);

        const auto nom        = c * k[idx] + coeff * phi[idx];
        const auto nom_grad   = c_grad * k[idx] + coeff * phi_grad[idx];
        const auto denom      = k[0] * phi[1] + k[1] * phi[0];
        const auto denom_grad = k[0] * phi_grad[1] + k[1] * phi_grad[0];

        return (nom_grad * denom - nom * denom_grad) / (denom * denom);
      };

    const auto calc_ga =
      [&](const VectorizedArrayType &ca, const Number ki, const Number c_0i) {
        return 0.5 * ki * std::pow(ca - c_0i, 2.);
      };

    const auto calc_mu = [&](const VectorizedArrayType &ca,
                             const Number               ki,
                             const Number c_0i) { return ki * (ca - c_0i); };



    // time loop: variables are phi_0, phi_1, c
    unsigned int counter = 0;
    for (double t = 0; counter++ < n_time_steps; t += dt)
      {
        // compute right-hand side vector
        matrix_free.template cell_loop<VectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, auto cells) {
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                eval.reinit(cell);
                eval.gather_evaluate(src,
                                     EvaluationFlags::values |
                                       EvaluationFlags::gradients);
                for (unsigned int q = 0; q < eval.n_q_points; ++q)
                  {
                    const auto value    = eval.get_value(q);
                    const auto gradient = eval.get_gradient(q);

                    // Field values
                    std::array<VectorizedArrayType, n_phi> phi = {
                      {value[0], value[1]}};
                    const auto c = value[2];

                    // Field gradients
                    const auto phi_grad =
                      std::array<Tensor<1, dim, VectorizedArrayType>, n_phi>{
                        {gradient[0], gradient[1]}};
                    const auto c_grad = gradient[2];

                    // Some auxiliary variables
                    const auto ca_0 = calc_ca(c, phi, 0);
                    const auto ca_1 = calc_ca(c, phi, 1);
                    const auto grad_ca_0 =
                      calc_grad_ca(c, c_grad, phi, phi_grad, 0);
                    const auto grad_ca_1 =
                      calc_grad_ca(c, c_grad, phi, phi_grad, 1);

                    // Chemical potential
                    const auto mu_0 = calc_mu(ca_0, k[0], c_0[0]);
                    const auto mu_1 = calc_mu(ca_1, k[1], c_0[1]);

                    const auto ga_0 = calc_ga(ca_0, k[0], c_0[0]);
                    const auto ga_1 = calc_ga(ca_1, k[1], c_0[1]);

                    // Number of active phases
                    const VectorizedArrayType zeroes(0.0), ones(1.0);

                    const auto nsz =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        phi[0], zeroes, ones, zeroes) +
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        phi[1], zeroes, ones, zeroes);

                    auto vvars0 = 2. * B * phi[0] * std::pow(phi[1], 2.) +
                                  ga_0 - mu_0 * ca_0;
                    auto vvars1 = 2. * B * phi[1] * std::pow(phi[0], 2.) +
                                  ga_1 - mu_1 * ca_1;

                    Tensor<1, n_comp, VectorizedArrayType> value_result;
                    value_result[0] = -L / nsz * (vvars0 - vvars1);
                    value_result[1] = -L / nsz * (vvars1 - vvars0);
                    value_result[2] = 0.;

                    Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                      gradient_result;

                    gradient_result[0] =
                      L / nsz * A * (phi_grad[1] - phi_grad[0]);
                    gradient_result[1] =
                      L / nsz * A * (phi_grad[0] - phi_grad[1]);
                    gradient_result[2] =
                      -D * (phi[0] * grad_ca_0 + phi[1] * grad_ca_1);

                    eval.submit_value(value_result, q);
                    eval.submit_gradient(gradient_result, q);
                  }
                eval.integrate_scatter(EvaluationFlags::values |
                                         EvaluationFlags::gradients,
                                       dst);
              }
          },
          dst,
          src,
          true);

        {
          ReductionControl     reduction_control;
          SolverCG<VectorType> solver(reduction_control);
          solver.solve(mass_matrix, incr, dst, PreconditionIdentity());

          std::cout << "it " << counter << ": " << reduction_control.last_step()
                    << std::endl;
        }

        // output_result(dst, t);
        // output_result(src, t);
        // output_result(incr, t);
        // throw std::runtime_error("STOP");

        for (unsigned int c = 0; c < n_comp; ++c)
          incr.block(c) *= dt;

        src += incr;

        if (counter % n_time_steps_output == 0)
          output_result(src, t);
      }
  }
};


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Test<2, 1>                       runner;
  runner.run();
}