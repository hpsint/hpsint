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
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;


namespace Preconditioners
{
  template <typename Number>
  class PreconditionerBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

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
      precondition_amg.initialize(op.get_system_matrix());
    }

  private:
    const Operator &op;

    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    TrilinosWrappers::PreconditionAMG                 precondition_amg;
  };
} // namespace Preconditioners


namespace LinearSolvers
{
  template <typename Number>
  class LinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    virtual unsigned int
    solve(VectorType &dst, const VectorType &src, const bool do_update) = 0;
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
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src, const bool do_update) override
    {
      if (do_update)
        preconditioner.do_update();

      unsigned int            max_iter = 100;
      ReductionControl        reduction_control(max_iter);
      SolverGMRES<VectorType> solver(reduction_control);
      solver.solve(op, dst, src, preconditioner);

      return reduction_control.last_step();
    }

    const Operator &op;
    Preconditioner &preconditioner;
  };
} // namespace LinearSolvers



namespace NonLinearSolvers
{
  struct NonLinearSolverStatistics
  {
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;
  };



  template <typename Number>
  class NonLinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    virtual NonLinearSolverStatistics
    solve(
      VectorType &       dst,
      bool const         update_preconditioner_linear_solver     = true,
      unsigned int const update_preconditioner_every_newton_iter = true) = 0;



    virtual NonLinearSolverStatistics
    solve(
      VectorType &       dst,
      VectorType const & rhs,
      bool const         update_preconditioner_linear_solver     = true,
      unsigned int const update_preconditioner_every_newton_iter = true) = 0;
  };



  struct NewtonSolverData
  {
    NewtonSolverData(const unsigned int max_iter = 100,
                     const double       abs_tol  = 1.e-20,
                     const double       rel_tol  = 1.e-5)
      : max_iter(max_iter)
      , abs_tol(abs_tol)
      , rel_tol(rel_tol)
    {}

    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;
  };



  template <typename VectorType,
            typename NonlinearOperator,
            typename SolverLinearizedProblem>
  class NewtonSolver
    : public NonLinearSolverBase<typename NonlinearOperator::value_type>
  {
  public:
    NewtonSolver(NonlinearOperator &      nonlinear_operator_in,
                 SolverLinearizedProblem &linear_solver_in,
                 const NewtonSolverData & solver_data_in = NewtonSolverData())
      : solver_data(solver_data_in)
      , nonlinear_operator(nonlinear_operator_in)
      , linear_solver(linear_solver_in)
    {
      nonlinear_operator.initialize_dof_vector(residual);
      nonlinear_operator.initialize_dof_vector(increment);
      nonlinear_operator.initialize_dof_vector(tmp);
    }

    NonLinearSolverStatistics
    solve(VectorType &       dst,
          bool const         update_preconditioner_linear_solver     = true,
          unsigned int const update_preconditioner_every_newton_iter = true)
    {
      VectorType rhs;
      return this->solve(dst,
                         rhs,
                         update_preconditioner_linear_solver,
                         update_preconditioner_every_newton_iter);
    }



    NonLinearSolverStatistics
    solve(VectorType &       dst,
          VectorType const & rhs,
          bool const         update_preconditioner_linear_solver     = true,
          unsigned int const update_preconditioner_every_newton_iter = true)
    {
      const bool constant_rhs = rhs.size() > 0;

      // evaluate residual using the given estimate of the solution
      nonlinear_operator.evaluate_nonlinear_residual(residual, dst);

      if (constant_rhs)
        residual -= rhs;

      double norm_r   = residual.l2_norm();
      double norm_r_0 = norm_r;

      // Accumulated linear iterations
      NonLinearSolverStatistics statistics;

#ifdef DEBUG_NORM
      std::cout << "NORM: " << std::flush;
#endif

      while (norm_r > this->solver_data.abs_tol &&
             norm_r / norm_r_0 > solver_data.rel_tol &&
             statistics.newton_iterations < solver_data.max_iter)
        {
#ifdef DEBUG_NORM
          std::cout << norm_r << " " << std::flush;
#endif
          // reset increment
          increment = 0.0;

          // multiply by -1.0 since the linearized problem is "LinearMatrix *
          // increment = - residual"
          residual *= -1.0;

          // solve linear problem
          nonlinear_operator.set_solution_linearization(dst);
          nonlinear_operator.evaluate_newton_step(dst);
          bool const do_update = update_preconditioner_linear_solver &&
                                 (statistics.newton_iterations %
                                    update_preconditioner_every_newton_iter ==
                                  0);
          statistics.linear_iterations +=
            linear_solver.solve(increment, residual, do_update);

          // damped Newton scheme
          double omega = 1.0; // damping factor
          double tau   = 0.1; // another parameter (has to be smaller than 1)
          double norm_r_tmp = 1.0; // norm of residual using temporary solution
          unsigned int n_iter_tmp = 0,
                       N_ITER_TMP_MAX =
                         100; // iteration counts for damping scheme

          do
            {
              // calculate temporary solution
              tmp = dst;
              tmp.add(omega, increment);


              // evaluate residual using the temporary solution
              nonlinear_operator.evaluate_nonlinear_residual(residual, tmp);
              if (constant_rhs)
                residual -= rhs;

              // calculate norm of residual (for temporary solution)
              norm_r_tmp = residual.l2_norm();

              // reduce step length
              omega = omega / 2.0;

              // increment counter
              n_iter_tmp++;
            }
          while (norm_r_tmp >= (1.0 - tau * omega) * norm_r &&
                 n_iter_tmp < N_ITER_TMP_MAX);

          AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                      ExcMessage("Damped Newton iteration did not converge. "
                                 "Maximum number of iterations exceeded!"));

          // update solution and residual
          dst    = tmp;
          norm_r = norm_r_tmp;

          // increment iteration counter
          ++statistics.newton_iterations;
        }

#ifdef DEBUG_NORM
      std::cout << std::endl;
#endif

      AssertThrow(
        norm_r <= this->solver_data.abs_tol ||
          norm_r / norm_r_0 <= solver_data.rel_tol,
        ExcMessage(
          "Newton solver failed to solve nonlinear problem to given tolerance. "
          "Maximum number of iterations exceeded!"));

      return statistics;
    }


  private:
    NewtonSolverData         solver_data;
    NonlinearOperator &      nonlinear_operator;
    SolverLinearizedProblem &linear_solver;

    VectorType residual, increment, tmp;
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
            int n_components,
            typename Number,
            typename VectorizedArrayType>
  class Operator
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    using FECellIntegrator =
      FEEvaluation<dim, -1, 0, n_components, Number, VectorizedArrayType>;

    Operator(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const AffineConstraints<Number> &                   constraints,
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
      , constraints(constraints)
      , free_energy(A, B)
      , mobility(Mvol, Mvap, Msurf, Mgb)
      , L(L)
      , kappa_c(kappa_c)
      , kappa_p(kappa_p)
    {}

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix_free.cell_loop(&Operator::do_vmult_range, this, dst, src, true);
    }

    void
    evaluate_nonlinear_residual(VectorType &dst, const VectorType &src) const
    {
      matrix_free.cell_loop(
        &Operator::do_evaluate_nonlinear_residual, this, dst, src, true);
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst);
    }

    void
    set_solution_linearization(const VectorType &src) const
    {
      (void)src; // TODO: linearization point is not used!?
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
      const unsigned n_cells             = matrix_free.n_cell_batches();
      const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

      nonlinear_values.reinit(n_cells, n_quadrature_points);
      nonlinear_mu.reinit(n_cells, n_quadrature_points);

      int dummy = 0;

      matrix_free.cell_loop(&Operator::do_evaluate_newton_step,
                            this,
                            dummy,
                            newton_step);
    }

    void
    set_timestep(double dt_new)
    {
      this->dt = dt_new;
    }

    void
    compute_inverse_diagonal(VectorType &diagonal) const
    {
      MatrixFreeTools::compute_diagonal(matrix_free,
                                        diagonal,
                                        &Operator::do_vmult_cell,
                                        this);
      for (auto &i : diagonal)
        i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      system_matrix.clear();

      const auto &dof_handler = this->matrix_free.get_dof_handler();

      TrilinosWrappers::SparsityPattern dsp(
        dof_handler.locally_owned_dofs(),
        dof_handler.get_triangulation().get_communicator());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
      dsp.compress();

      system_matrix.reinit(dsp);

      MatrixFreeTools::compute_matrix(matrix_free,
                                      constraints,
                                      system_matrix,
                                      &Operator::do_vmult_cell,
                                      this);

      return system_matrix;
    }


  private:
    void
    do_vmult_kernel(FECellIntegrator &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

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
            L * free_energy.d2f_dcdetai(c, etas, 0) * phi.get_value(q)[0] +
            L * free_energy.d2f_detai2(c, etas, 0) * phi.get_value(q)[2] +
            L * free_energy.d2f_detaidetaj(c, etas, 0, 1) * phi.get_value(q)[3];
          value_result[3] =
            phi.get_value(q)[3] / dt +
            L * free_energy.d2f_dcdetai(c, etas, 1) * phi.get_value(q)[0] +
            L * free_energy.d2f_detaidetaj(c, etas, 1, 0) *
              phi.get_value(q)[2] +
            L * free_energy.d2f_detai2(c, etas, 1) * phi.get_value(q)[3];

          Tensor<1, n_components, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          gradient_result[0] =
            mobility.M(c, etas) * phi.get_gradient(q)[1] +
            mobility.dM_dc(c, etas) * mu_grad * phi.get_value(q)[0] +
            mobility.dM_detai(c, etas, 0) * mu_grad * phi.get_value(q)[2] +
            mobility.dM_detai(c, etas, 1) * mu_grad * phi.get_value(q)[3];
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
          gradient_result[2] = L * kappa_p * phi.get_gradient(q)[2];
          gradient_result[3] = L * kappa_p * phi.get_gradient(q)[3];

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

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
      FECellIntegrator phi(matrix_free);

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

    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator phi_old(matrix_free);
      FECellIntegrator phi(matrix_free);

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
      FECellIntegrator phi(matrix_free);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values_plain(src);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              nonlinear_values(cell, q) = phi.get_value(q);
              nonlinear_mu(cell, q)     = phi.get_gradient(q)[1];
            }
        }
    }


    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const AffineConstraints<Number> &                   constraints;

    const FreeEnergy free_energy;
    const Mobility   mobility;
    const double     L;
    const double     kappa_c;
    const double     kappa_p;

    double dt;

    mutable VectorType old_solution;

    Table<2, dealii::Tensor<1, n_components, VectorizedArrayType>>
                                                          nonlinear_values;
    Table<2, dealii::Tensor<1, dim, VectorizedArrayType>> nonlinear_mu;

    mutable TrilinosWrappers::SparseMatrix system_matrix;
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
      Operator<dim, number_of_components, Number, VectorizedArrayType>;

    // geometry
    static constexpr double diameter        = 15.0;
    static constexpr double interface_width = 2.0;
    static constexpr double boundary_factor = 1.0;

    // mesh
    static constexpr unsigned int elements_per_interface = 4;

    // time discretization
    static constexpr double t_end                = 100;
    static constexpr double dt_deseride          = 0.001;
    static constexpr double dt_max               = 1e2 * dt_deseride;
    static constexpr double dt_min               = 1e-2 * dt_deseride;
    static constexpr double dt_increment         = 1.2;
    static constexpr double output_time_interval = 0.1;

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

    ConditionalOStream                        pcout;
    parallel::distributed::Triangulation<dim> tria;
    FESystem<dim>                             fe;
    MappingQ<dim>                             mapping;
    QGauss<dim>                               quad;
    DoFHandler<dim>                           dof_handler;
    AffineConstraints<Number> constraint; // TODO: currently no constraints are
                                          // applied

    InitialValues<dim> initial_solution;

    Problem(const unsigned int fe_degree, const unsigned int n_points_1D)
      : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , tria(MPI_COMM_WORLD)
      , fe(FE_Q<dim>{fe_degree}, number_of_components)
      , mapping(1)
      , quad(n_points_1D)
      , dof_handler(tria)
      , initial_solution(x01,
                         x02,
                         y0,
                         r0,
                         interface_width,
                         number_of_components,
                         is_accumulative)
    {
      // create mesh
      create_mesh(tria,
                  domain_width,
                  domain_height,
                  interface_width,
                  elements_per_interface);

      // distribute dofs
      dof_handler.distribute_dofs(fe);
    }

    void
    run()
    {
      // setup MatrixFree ...
      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags = update_values | update_gradients;

      MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
      matrix_free.reinit(
        mapping, dof_handler, constraint, quad, additional_data);

      // ... non-linear operator
      NonLinearOperator nonlinear_operator(matrix_free,
                                           constraint,
                                           A,
                                           B,
                                           Mvol,
                                           Mvap,
                                           Msurf,
                                           Mgb,
                                           L,
                                           kappa_c,
                                           kappa_p);

      // ... preconditioner
      std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
        preconditioner;

      if (true)
        preconditioner = std::make_unique<
          Preconditioners::InverseDiagonalMatrix<NonLinearOperator>>(
          nonlinear_operator);
      else
        preconditioner =
          std::make_unique<Preconditioners::AMG<NonLinearOperator>>(
            nonlinear_operator);

      // ... linear solver
      std::unique_ptr<LinearSolvers::LinearSolverBase<Number>> linear_solver;

      if (true)
        linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
          NonLinearOperator,
          Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                        *preconditioner);

      // ... non-linear Newton solver
      std::unique_ptr<NonLinearSolvers::NonLinearSolverBase<Number>>
        non_linear_solver;

      if (true)
        non_linear_solver = std::make_unique<NonLinearSolvers::NewtonSolver<
          VectorType,
          NonLinearOperator,
          LinearSolvers::LinearSolverBase<Number>>>(nonlinear_operator,
                                                    *linear_solver);

      // set initial condition
      VectorType solution;

      nonlinear_operator.initialize_dof_vector(solution);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               initial_solution,
                               solution);

      double time_last_output = 0;
      output_result(solution, time_last_output);

      // run time loop
      for (double t = 0, dt = dt_deseride; t <= t_end;)
        {
          nonlinear_operator.set_timestep(dt);
          nonlinear_operator.set_previous_solution(solution);
          nonlinear_operator.evaluate_newton_step(solution);

          preconditioner->do_update();

          bool has_converged = false;

          try
            {
              const auto statistics = non_linear_solver->solve(solution);

              has_converged = true;

              pcout << "t = " << t << ", dt = " << dt << ":"
                    << " solved in " << statistics.newton_iterations
                    << " Newton iterations and " << statistics.linear_iterations
                    << " linear iterations" << std::endl;

              if (std::abs(t - t_end) > 1e-9)
                {
                  if (statistics.newton_iterations <
                        desirable_newton_iterations &&
                      statistics.linear_iterations <
                        desirable_linear_iterations)
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

              AssertThrow(
                dt > dt_min,
                ExcMessage("Minimum timestep size exceeded, solution failed!"));
            }

          if (has_converged && t > output_time_interval + time_last_output)
            {
              time_last_output = t;
              output_result(solution, time_last_output);
            }
        }
    }

  private:
    void
    create_mesh(parallel::distributed::Triangulation<dim> &tria,
                const double                               domain_width,
                const double                               domain_height,
                const double                               interface_width,
                const unsigned int elements_per_interface = 4)
    {
      const unsigned int initial_ny = 10;
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
    output_result(const VectorType &solution, const double t)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      std::vector<std::string> names{"c", "mu", "eta1", "eta2"};
      data_out.add_data_vector(solution, names);

      solution.update_ghost_values();
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

  Sintering::Problem<2> runner(1, 2);
  runner.run();
}