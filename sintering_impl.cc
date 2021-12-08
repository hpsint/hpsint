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
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/sundials/kinsol.h>

#include "include/newton.h"

//#define IDENTITY
#define NEWTON

using namespace dealii;

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
        double int_pos  = (rad - rm + interface_width / 2.0) / interface_width;

        c = outvalue +
            (invalue - outvalue) * (1.0 + std::cos(int_pos * M_PI)) / 2.0;
        // c = 0.5 - 0.5 * std::sin(M_PI * (rad - rm) / interface_width);
      }

    return c;
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
              {
                Tensor<1, n_components, VectorizedArrayType> value_result;
                for (unsigned int i = 0; i < n_components; i++)
                  {
                    value_result[i] = phi.get_value(q)[i];
                  }
                phi.submit_value(value_result, q);
              }
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

class Mobility
{
protected:
  double Mvol;
  double Mvap;
  double Msurf;
  double Mgb;

public:
  Mobility(const double Mvol,
           const double Mvap,
           const double Msurf,
           const double Mgb)
    : Mvol(Mvol)
    , Mvap(Mvap)
    , Msurf(Msurf)
    , Mgb(Mgb)
  {}

  auto
  M(const auto &c, const std::vector<auto> &etas) const
  {
    auto cl = c;
    std::for_each(cl.begin(), cl.end(), [](auto &val) {
      val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
    });

    std::remove_const_t<std::remove_reference_t<decltype(c)>> etaijSum = 0.0;
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

    auto phi = cl * cl * cl * (10.0 - 15.0 * cl + 6.0 * cl * cl);
    std::for_each(phi.begin(), phi.end(), [](auto &val) {
      val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
    });

    auto M = Mvol * phi + Mvap * (1.0 - phi) + Msurf * cl * (1.0 - cl) +
             Mgb * etaijSum;

    return M;
  }

  auto
  dM_dc(const auto &c, const std::vector<auto> &etas) const
  {
    (void)etas;

    auto cl = c;
    std::for_each(cl.begin(), cl.end(), [](auto &val) {
      val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
    });

    auto dphidc = 30.0 * cl * cl * (1.0 - 2.0 * cl + cl * cl);
    auto dMdc   = Mvol * dphidc - Mvap * dphidc + Msurf * (1.0 - 2.0 * cl);

    return dMdc;
  }

  auto
  dM_detai(const auto &             c,
           const std::vector<auto> &etas,
           unsigned int             index_i) const
  {
    std::remove_const_t<std::remove_reference_t<decltype(c)>> etajSum = 0;
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

  auto
  f(const auto &c, const std::vector<auto> &etas) const
  {
    std::remove_const_t<std::remove_reference_t<decltype(c)>> initial = 0.0;

    auto etaPower2Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b;
      });
    auto etaPower3Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b * b;
      });

    return A * std::pow(c, 2.0) * std::pow(-c + 1.0, 2.0) +
           B * (std::pow(c, 2.0) + (-6.0 * c + 6.0) * etaPower2Sum -
                (-4.0 * c + 8.0) * etaPower3Sum +
                3.0 * std::pow(etaPower2Sum, 2.0));
  }

  auto
  df_dc(const auto &c, const std::vector<auto> &etas) const
  {
    std::remove_const_t<std::remove_reference_t<decltype(c)>> initial = 0.0;

    auto etaPower2Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b;
      });
    auto etaPower3Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b * b;
      });

    return A * std::pow(c, 2.0) * (2.0 * c - 2.0) +
           2.0 * A * c * std::pow(-c + 1.0, 2.0) +
           B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
  }

  auto
  df_detai(const auto &             c,
           const std::vector<auto> &etas,
           unsigned int             index_i) const
  {
    std::remove_const_t<std::remove_reference_t<decltype(c)>> initial = 0.0;

    auto etaPower2Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b;
      });

    auto &etai = etas[index_i];

    return B * (3.0 * std::pow(etai, 2.0) * (4.0 * c - 8.0) +
                2.0 * etai * (-6.0 * c + 6.0) + 12.0 * etai * (etaPower2Sum));
  }

  auto
  d2f_dc2(const auto &c, const std::vector<auto> &etas) const
  {
    (void)etas;

    return 2.0 * A * std::pow(c, 2.0) + 4.0 * A * c * (2.0 * c - 2.0) +
           2.0 * A * std::pow(-c + 1.0, 2.0) + 2.0 * B;
  }

  auto
  d2f_dcdetai(const auto &             c,
              const std::vector<auto> &etas,
              unsigned int             index_i) const
  {
    (void)c;

    auto &etai = etas[index_i];

    return B * (12.0 * std::pow(etai, 2.0) - 12.0 * etai);
  }

  auto
  d2f_detai2(const auto &             c,
             const std::vector<auto> &etas,
             unsigned int             index_i) const
  {
    std::remove_const_t<std::remove_reference_t<decltype(c)>> initial = 0.0;
    auto                                                      etaPower2Sum =
      std::accumulate(etas.begin(), etas.end(), initial, [](auto a, auto b) {
        return std::move(a) + b * b;
      });

    auto &etai = etas[index_i];

    return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                24.0 * std::pow(etai, 2.0) + 12.0 * etaPower2Sum);
  }

  auto
  d2f_detaidetaj(const auto &             c,
                 const std::vector<auto> &etas,
                 unsigned int             index_i,
                 unsigned int             index_j) const
  {
    (void)c;
    auto &etai = etas[index_i];
    auto &etaj = etas[index_j];

    return 24.0 * B * etai * etaj;
  }
};

template <int dim,
          int degree,
          int n_points_1D,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
class Sintering
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  using value_type  = Number;
  using vector_type = VectorType;

  Sintering(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
            const double                                        A,
            const double                                        B,
            const double                                        Mvol,
            const double                                        Mvap,
            const double                                        Msurf,
            const double                                        Mgb,
            const double                                        L,
            const double                                        kappa_c,
            const double                                        kappa_p)
    : matrix_free(matrix_free)
    , free_energy(A, B)
    , mobility(Mvol, Mvap, Msurf, Mgb)
    , L(L)
    , kappa_c(kappa_c)
    , kappa_p(kappa_p)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        FEEvaluation<dim,
                     degree,
                     n_points_1D,
                     n_components,
                     Number,
                     VectorizedArrayType>
          phi(matrix_free);
          
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src,
                                EvaluationFlags::EvaluationFlags::values|
                                EvaluationFlags::EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                auto &c    = nonlinear_values(cell, q)[0];
                auto &eta1 = nonlinear_values(cell, q)[2];
                auto &eta2 = nonlinear_values(cell, q)[3];

                std::vector etas{eta1, eta2};

                auto &mu_grad = nonlinear_mu(cell, q);

                Tensor<1, n_components, VectorizedArrayType> value_result;

                value_result[0] = phi.get_value(q)[0] / dt;
                value_result[1] =
                  -phi.get_value(q)[1] +
                  free_energy.d2f_dc2(c, etas) * phi.get_value(q)[0] +
                  free_energy.d2f_dcdetai(c, etas, 0) * phi.get_value(q)[2] +
                  free_energy.d2f_dcdetai(c, etas, 1) * phi.get_value(q)[3];
                value_result[2] =
                  phi.get_value(q)[2] / dt +
                  L * free_energy.d2f_dcdetai(c, etas, 0) *
                    phi.get_value(q)[0] +
                  L * free_energy.d2f_detai2(c, etas, 0) * phi.get_value(q)[2] +
                  L * free_energy.d2f_detaidetaj(c, etas, 0, 1) *
                    phi.get_value(q)[3];
                value_result[3] =
                  phi.get_value(q)[3] / dt +
                  L * free_energy.d2f_dcdetai(c, etas, 1) *
                    phi.get_value(q)[0] +
                  L * free_energy.d2f_detaidetaj(c, etas, 1, 0) *
                    phi.get_value(q)[2] +
                  L * free_energy.d2f_detai2(c, etas, 1) * phi.get_value(q)[3];

                Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
                  gradient_result;

                gradient_result[0] =
                  mobility.M(c, etas) * phi.get_gradient(q)[1] +
                  mobility.dM_dc(c, etas) * mu_grad * phi.get_value(q)[0] +
                  mobility.dM_detai(c, etas, 0) * mu_grad *
                    phi.get_value(q)[2] +
                  mobility.dM_detai(c, etas, 1) * mu_grad * phi.get_value(q)[3];
                gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
                gradient_result[2] = L * kappa_p * phi.get_gradient(q)[2];
                gradient_result[3] = L * kappa_p * phi.get_gradient(q)[3];

                phi.submit_value(value_result, q);
                phi.submit_gradient(gradient_result, q);
              }
            phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values|
                                  EvaluationFlags::EvaluationFlags::gradients,
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
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
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
          
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi_old.reinit(cell);
            phi.reinit(cell);

            phi.gather_evaluate(src,
                                EvaluationFlags::EvaluationFlags::values|
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

                std::vector etas{eta1, eta2};

                auto &c_old    = val_old[0];
                auto &eta1_old = val_old[2];
                auto &eta2_old = val_old[3];

                Tensor<1, n_components, VectorizedArrayType> value_result;

                value_result[0] = (c - c_old) / dt;
                value_result[1] = -mu + free_energy.df_dc(c, etas);
                value_result[2] =
                  (eta1 - eta1_old) / dt + L * free_energy.df_detai(c, etas, 0);
                value_result[3] =
                  (eta2 - eta2_old) / dt + L * free_energy.df_detai(c, etas, 1);

                Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
                  gradient_result;

                gradient_result[0] = mobility.M(c, etas) * grad[1];
                gradient_result[1] = kappa_c * grad[0];
                gradient_result[2] = L * kappa_p * grad[2];
                gradient_result[3] = L * kappa_p * grad[3];

                phi.submit_value(value_result, q);
                phi.submit_gradient(gradient_result, q);
              }
            phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values|
                                  EvaluationFlags::EvaluationFlags::gradients,
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
    (void)src;
  }

  void
  set_previous_solution(const VectorType &src) const
  {
    this->old_solution = src;
  }

  const VectorType &
  get_previous_solution() const
  {
    return this->old_solution;
  }

  void
  evaluate_newton_step(const VectorType &newton_step)
  {
    const unsigned int n_cells = matrix_free.n_cell_batches();

    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi(matrix_free);

    nonlinear_values.reinit(n_cells, phi.n_q_points);
    nonlinear_mu.reinit(n_cells, phi.n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values_plain(newton_step);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            nonlinear_values(cell, q) = phi.get_value(q);
            nonlinear_mu(cell, q)     = phi.get_gradient(q)[1];
          }
      }
  }

  void
  set_timestep(double dt_new)
  {
    this->dt = dt_new;
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
  const FreeEnergy                                    free_energy;
  const Mobility                                      mobility;
  const double                                        L;
  const double                                        kappa_c;
  const double                                        kappa_p;

  double dt;

  mutable VectorType old_solution;

  Table<2, dealii::Tensor<1, n_components, VectorizedArrayType>>
                                                        nonlinear_values;
  Table<2, dealii::Tensor<1, dim, VectorizedArrayType>> nonlinear_mu;
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
class SolverGMRESWrapper
{
public:
  using VectorType = typename Operator::vector_type;

  SolverGMRESWrapper(const Operator &op, const Preconditioner &preconditioner)
    : op(op)
    , preconditioner(preconditioner)
  {}

  unsigned int
  solve(VectorType &dst, const VectorType &src, const bool do_update)
  {
    (void)do_update; // no preconditioner is used

    unsigned int            max_iter = 100;
    ReductionControl        reduction_control(max_iter);
    SolverGMRES<VectorType> solver(reduction_control);
    solver.solve(op, dst, src, preconditioner);

    return reduction_control.last_step();
  }

  const Operator &      op;
  const Preconditioner &preconditioner;
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

    unsigned int         max_iter = 200;
    ReductionControl     reduction_control(max_iter);
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
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  void
  run()
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // geometry
    const double diameter        = 15.0;
    const double interface_width = 2.0;
    const double boundary_factor = 1.0;

    // mesh
    const unsigned int elements_per_interface = 4;

    // time discretization
    const double t_end                = 100;
    double       dt                   = 0.001;
    const double dt_max               = 1e2 * dt;
    const double dt_min               = 1e-2 * dt;
    const double dt_increment         = 1.2;
    const double output_time_interval = 0.1;

    // desirable number of newton iterations
    const unsigned int desirable_newton_iterations = 5;
    const unsigned int desirable_linear_iterations = 100;

    //  model parameters
    const double A       = 16;
    const double B       = 1;
    const double Mvol    = 1e-2;
    const double Mvap    = 1e-10;
    const double Msurf   = 4;
    const double Mgb     = 0.4;
    const double L       = 1;
    const double kappa_c = 1;
    const double kappa_p = 0.5;

    // components number
    constexpr unsigned int number_of_components = 4;

    // Create mesh
    double domain_width  = 2 * diameter + boundary_factor * diameter;
    double domain_height = 1 * diameter + boundary_factor * diameter;
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    create_mesh(tria,
               domain_width,
               domain_height,
               interface_width,
               elements_per_interface);

    FESystem<dim>   fe(FE_Q<dim>{fe_degree}, number_of_components);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ<dim> mapping(1);

    QGauss<dim> quad(n_points_1D);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

    VectorType solution;

    matrix_free.initialize_dof_vector(solution);

    // Initial values
    double x01             = domain_width / 2. - diameter / 2.;
    double x02             = domain_width / 2. + diameter / 2.;
    double y0              = domain_height / 2.;
    double r0              = diameter / 2.;
    bool   is_accumulative = false;
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim>(x01,
                                                x02,
                                                y0,
                                                r0,
                                                interface_width,
                                                number_of_components,
                                                is_accumulative),
                             solution);

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      std::vector<std::string> names{"c", "mu", "eta1", "eta2"};
      data_out.add_data_vector(solution, names);

      solution.update_ghost_values();
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;

      std::cout << "Outputing at t = " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };

    double time_last_output = 0;
    output_result(time_last_output);

    using Operator = Sintering<dim,
                               fe_degree,
                               n_points_1D,
                               number_of_components,
                               Number,
                               VectorizedArrayType>;

#ifdef IDENTITY
    using Preconditioner = PreconditionIdentity;
#else
    using PreconditionerOperator = MassMatrix<dim,
                                              fe_degree,
                                              n_points_1D,
                                              number_of_components,
                                              Number,
                                              VectorizedArrayType>;
    using Preconditioner         = InverseMassMatrix<PreconditionerOperator>;
#endif
    // using LinearSolver = SolverCGWrapper<Operator, Preconditioner>;
    using LinearSolver = SolverGMRESWrapper<Operator, Preconditioner>;

    Operator nonlinear_operator(
      matrix_free, A, B, Mvol, Mvap, Msurf, Mgb, L, kappa_c, kappa_p);

#ifdef IDENTITY
    Preconditioner preconditioner;
#else
    PreconditionerOperator precondition_operator(matrix_free);
    Preconditioner         preconditioner(precondition_operator);
#endif

    LinearSolver linear_solver(nonlinear_operator, preconditioner);

    NewtonSolver<VectorType, Operator, LinearSolver> newton_solver(
      nonlinear_operator, linear_solver);

    // time loop
    for (double t = 0; t <= t_end;)
      {
        nonlinear_operator.set_timestep(dt);
        nonlinear_operator.set_previous_solution(solution);
        nonlinear_operator.evaluate_newton_step(solution);

        bool has_converged = false;

        try
          {
            const auto statistics = newton_solver.solve(solution);

            has_converged = true;

            pcout << "t = " << t << ", dt = " << dt << ":"
                  << " solved in " << statistics.newton_iterations
                  << " Newton iterations and " << statistics.linear_iterations
                  << " linear iterations" << std::endl;

            if (std::abs(t - t_end) > 1e-9)
              {
                if (statistics.newton_iterations <
                      desirable_newton_iterations &&
                    statistics.linear_iterations < desirable_linear_iterations)
                  {
                    dt *= dt_increment;
                    pcout << "Increasing timestep, dt = " << dt << std::endl;

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
        catch (...)
          {
            dt *= 0.5;
            pcout << "Solver diverged, reducing timestep, dt = " << dt
                  << std::endl;

            solution = nonlinear_operator.get_previous_solution();

            AssertThrow(dt > dt_min,
                        ExcMessage(
                          "Minimum timestep size exceeded, solution failed!"));
          }

        if (has_converged && t > output_time_interval + time_last_output)
          {
            time_last_output = t;
            output_result(time_last_output);
          }
      }
  }

private:
  void
  create_mesh(parallel::distributed::Triangulation<dim> &tria,
             const double                                     domain_width,
             const double                                     domain_height,
             const double                                     interface_width,
             const unsigned int elements_per_interface = 4)
  {
    const unsigned int initial_ny = 10;
    const unsigned int initial_nx = int(domain_width / domain_height * initial_ny);

    const unsigned int n_refinements = int(std::round(std::log2(
      elements_per_interface / interface_width * domain_height / initial_ny)));

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
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Test<2, 1, 2>                    runner;
  runner.run();
}