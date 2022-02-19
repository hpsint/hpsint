#pragma once

#include <pf-applications/lac/preconditioners.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliard
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliard<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorCahnHilliard(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          1,
          "cahn_hilliard_op")
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 2;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      static_assert(n_grains != -1);
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy         = this->op.get_data().free_energy;
      const auto &mobility            = this->op.get_data().mobility;
      const auto &kappa_c             = this->op.get_data().kappa_c;
      const auto &dt                  = this->op.get_dt();
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      // TODO: 1) allow std::array again and 2) allocate less often in the
      // case of std::vector
      std::array<const VectorizedArrayType *, n_grains> etas;
      std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
        etas_grad;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c       = val[0];
          const auto &c_grad  = grad[0];
          const auto &mu_grad = grad[1];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

          Tensor<1, n_comp, VectorizedArrayType> value_result;
          Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
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

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliardA
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliardA<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorCahnHilliardA(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliardA<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 1;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      static_assert(n_grains != -1);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &dt                  = this->op.get_dt();
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      // TODO: see above
      std::array<const VectorizedArrayType *, n_grains> etas;
      std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
        etas_grad;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c       = val[0];
          const auto &c_grad  = grad[0];
          const auto &mu_grad = grad[1];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

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

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliardB
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliardB<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorCahnHilliardB(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliardB<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 1;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      // TODO: see above
      std::array<const VectorizedArrayType *, n_grains> etas;
      std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
        etas_grad;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c      = val[0];
          const auto &c_grad = grad[0];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(value * 0.0, q); // TODO
          phi.submit_gradient(mobility.M(c, etas, c_grad, etas_grad) * gradient,
                              q);
        }
    }

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliardC
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliardC<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorCahnHilliardC(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliardC<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 1;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy      = this->op.get_data().free_energy;
      const auto &kappa_c          = this->op.get_data().kappa_c;
      const auto &nonlinear_values = this->op.get_nonlinear_values();

      // TODO: see above
      std::array<const VectorizedArrayType *, n_grains> etas;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val = nonlinear_values[cell][q];

          const auto &c = val[0];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig] = &val[2 + ig];
            }

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(free_energy.d2f_dc2(c, etas) * value, q);
          phi.submit_gradient(kappa_c * gradient, q);
        }
    }

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliardD
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliardD<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorCahnHilliardD(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliardD<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3 /*TODO*/)
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 1;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);

          phi.submit_value(-value, q);
          phi.submit_gradient(gradient * 0.0, q); // TODO
        }
    }

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorAllenCahn
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          OperatorAllenCahn<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorAllenCahn(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          2,
          "allen_cahn_op")
      , op(op)
    {}

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      static_assert(n_comp == n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy      = this->op.get_data().free_energy;
      const auto &L                = this->op.get_data().L;
      const auto &kappa_p          = this->op.get_data().kappa_p;
      const auto &dt               = this->op.get_dt();
      const auto &nonlinear_values = this->op.get_nonlinear_values();

      const auto dt_inv = 1.0 / dt;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val = nonlinear_values[cell][q];

          const auto &c = val[0];

          std::array<const VectorizedArrayType *, n_grains> etas;

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            etas[ig] = &val[2 + ig];

          Tensor<1, n_comp, VectorizedArrayType> value_result;
          Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[ig] =
                phi.get_value(q)[ig] * dt_inv +
                L * free_energy.d2f_detai2(c, etas, ig) * phi.get_value(q)[ig];

              gradient_result[ig] = L * kappa_p * phi.get_gradient(q)[ig];

              for (unsigned int jg = 0; jg < n_grains; ++jg)
                {
                  if (ig != jg)
                    {
                      value_result[ig] +=
                        L * free_energy.d2f_detaidetaj(c, etas, ig, jg) *
                        phi.get_value(q)[jg];
                    }
                }
            }

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

  private:
    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorAllenCahnHelmholtz
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorAllenCahnHelmholtz<dim, Number, VectorizedArrayType>>
  {
  public:
    using VectorType = typename OperatorBase<
      dim,
      Number,
      VectorizedArrayType,
      OperatorAllenCahnHelmholtz<dim, Number, VectorizedArrayType>>::VectorType;

    OperatorAllenCahnHelmholtz(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          OperatorAllenCahnHelmholtz<dim, Number, VectorizedArrayType>>(
          matrix_free,
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

    template <int n_comp, int>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &) const
    {
      AssertThrow(false, ExcNotImplemented());
    }

  private:
    void
    do_vmult_cell_mass(
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate(EvaluationFlags::EvaluationFlags::values);
    }
    void
    do_vmult_cell_laplace(
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> &phi) const
    {
      const auto &L       = this->op.get_data().L;
      const auto &kappa_p = this->op.get_data().kappa_p;

      phi.evaluate(EvaluationFlags::EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_gradient(L * kappa_p * phi.get_gradient(q), q);

      phi.integrate(EvaluationFlags::EvaluationFlags::gradients);
    }

    const SinteringOperator<dim, Number, VectorizedArrayType> &op;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class InverseDiagonalMatrixAllenCahnHelmholtz
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using Operator =
      OperatorAllenCahnHelmholtz<dim, Number, VectorizedArrayType>;
    using VectorType = typename Operator::VectorType;
    using BlockVectorType =
      typename Preconditioners::PreconditionerBase<Number>::BlockVectorType;

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

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < diag.locally_owned_size(); ++i)
        for (unsigned int c = 0; c < op.n_components(); ++c)
          dst_ptr[i * op.n_components() + c] =
            diag_ptr[i] * src_ptr[i * op.n_components() + c];
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      AssertDimension(dst.n_blocks(), 1);
      AssertDimension(src.n_blocks(), 1);

      this->vmult(dst.block(0), src.block(0));
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



  template <int dim, typename Number, typename VectorizedArrayType>
  class BlockPreconditioner2
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;
    using BlockVectorType =
      typename Preconditioners::PreconditionerBase<Number>::BlockVectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner2(
      const SinteringOperator<dim, Number, VectorizedArrayType> &op,
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const BlockPreconditioner2Data &                           data = {})
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
        VectorTools::split_up_fast(src,
                                   src_0,
                                   src_1,
                                   operator_1.n_components());

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
        VectorTools::merge_fast(dst_0, dst_1, dst, operator_1.n_components());

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      VectorType dst_, src_;                         // TODO
      matrix_free.initialize_dof_vector(dst_, 0);    // TODO
      matrix_free.initialize_dof_vector(src_, 0);    // TODO
      VectorTools::merge_components_fast(src, src_); // TODO

      this->vmult(dst_, src_);

      VectorTools::split_up_components_fast(dst_, dst); // TODO
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

    OperatorCahnHilliard<dim, Number, VectorizedArrayType> operator_0;
    OperatorAllenCahn<dim, Number, VectorizedArrayType>    operator_1;
    OperatorAllenCahnHelmholtz<dim, Number, VectorizedArrayType>
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



  template <int dim, typename Number, typename VectorizedArrayType>
  class BlockPreconditioner3
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;
    using BlockVectorType =
      typename Preconditioners::PreconditionerBase<Number>::BlockVectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner3(
      const SinteringOperator<dim, Number, VectorizedArrayType> &op,
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const BlockPreconditioner3Data &                           data = {})
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
        VectorTools::split_up_fast(
          src, src_0, src_1, src_2, operator_2.n_components());

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
        VectorTools::merge_fast(
          dst_0, dst_1, dst_2, dst, operator_2.n_components());

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, dst_2, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    vmult(BlockVectorType &, const BlockVectorType &) const override
    {
      Assert(false, ExcNotImplemented());
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

    OperatorCahnHilliardA<dim, Number, VectorizedArrayType> operator_0;
    OperatorCahnHilliardB<dim, Number, VectorizedArrayType> block_ch_b;
    OperatorCahnHilliardC<dim, Number, VectorizedArrayType> block_ch_c;
    OperatorCahnHilliardD<dim, Number, VectorizedArrayType> operator_1;

    OperatorAllenCahn<dim, Number, VectorizedArrayType> operator_2;

    mutable VectorType dst_0, dst_1, dst_2;
    mutable VectorType src_0, src_1, src_2;

    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_1, preconditioner_2;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    BlockPreconditioner3Data data;
  };


  template <int dim, typename Number, typename VectorizedArrayType>
  class MassMatrix
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          MassMatrix<dim, Number, VectorizedArrayType>>
  {
  public:
    MassMatrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     MassMatrix<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3,
          "mass_matrix_op")
    {}

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      static_assert(n_grains == -1);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(
            typename FECellIntegrator<dim,
                                      n_comp,
                                      Number,
                                      VectorizedArrayType>::gradient_type(),
            q);
        }
    }
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorCahnHilliardHelmholtz
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType>>
  {
  public:
    using VectorType = typename OperatorBase<
      dim,
      Number,
      VectorizedArrayType,
      OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType>>::
      VectorType;

    OperatorCahnHilliardHelmholtz(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
      : OperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType>>(
          matrix_free,
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

    unsigned int
    n_grains() const
    {
      return op.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return 1;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      const auto sqrt_delta = this->get_sqrt_delta();
      const auto dt         = get_dt();

      // TODO: see above
      std::array<const VectorizedArrayType *, n_grains> etas;
      std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
        etas_grad;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c      = val[0];
          const auto &c_grad = grad[0];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

          const auto value    = phi.get_value(q);
          const auto gradient = phi.get_gradient(q);
          const auto epsilon  = dt * mobility.M(c, etas, c_grad, etas_grad);

          phi.submit_value(value, q);
          phi.submit_gradient(std::sqrt(std::abs(sqrt_delta * epsilon)) *
                                gradient,
                              q);
        }
    }

  private:
    template <bool use_mobility>
    void
    do_vmult_cell_laplace(
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::gradients);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &mobility            = this->op.get_data().mobility;
      const auto &nonlinear_values    = this->op.get_nonlinear_values();
      const auto &nonlinear_gradients = this->op.get_nonlinear_gradients();

      const auto sqrt_delta = this->get_sqrt_delta();
      const auto dt         = get_dt();

      // TODO: see above
      std::vector<const VectorizedArrayType *> etas(this->op.n_grains());
      std::vector<const Tensor<1, dim, VectorizedArrayType> *> etas_grad(
        this->op.n_grains());

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c      = val[0];
          const auto &c_grad = grad[0];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

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

    const SinteringOperator<dim, Number, VectorizedArrayType> &op;

    mutable VectorType epsilon;

    mutable double dt;
  };



  struct BlockPreconditioner3CHData
  {
    std::string block_0_preconditioner = "AMG";
    std::string block_2_preconditioner = "InverseDiagonalMatrix";
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class BlockPreconditioner3CHOperator : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    BlockPreconditioner3CHOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const SinteringOperator<dim, Number, VectorizedArrayType> &op)
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
    OperatorCahnHilliardA<dim, Number, VectorizedArrayType> operator_a;
    OperatorCahnHilliardB<dim, Number, VectorizedArrayType> operator_b;
    OperatorCahnHilliardC<dim, Number, VectorizedArrayType> operator_c;
    OperatorCahnHilliardD<dim, Number, VectorizedArrayType> operator_d;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class BlockPreconditioner3CHPreconditioner
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;


    BlockPreconditioner3CHPreconditioner(
      const OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType>
        &                                                 operator_0,
      const MassMatrix<dim, Number, VectorizedArrayType> &mass_matrix,
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
    const OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType>
      &operator_0;

    const MassMatrix<dim, Number, VectorizedArrayType> &mass_matrix;

    const std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      &preconditioner_0;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class BlockPreconditioner3CH
    : public Preconditioners::PreconditionerBase<Number>
  {
  public:
    using VectorType =
      typename Preconditioners::PreconditionerBase<Number>::VectorType;
    using BlockVectorType =
      typename Preconditioners::PreconditionerBase<Number>::BlockVectorType;

    using value_type  = Number;
    using vector_type = VectorType;

    BlockPreconditioner3CH(
      const SinteringOperator<dim, Number, VectorizedArrayType> &op,
      const MatrixFree<dim, Number, VectorizedArrayType> &       matrix_free,
      const std::vector<const AffineConstraints<Number> *> &     constraints,
      const BlockPreconditioner3CHData &                         data = {})
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
        VectorTools::split_up_fast(
          src, src_0, src_1, src_2, operator_2.n_components());

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
              BlockPreconditioner3CHOperator<dim, Number, VectorizedArrayType>,
              BlockPreconditioner3CHPreconditioner<dim,
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
        VectorTools::merge_fast(
          dst_0, dst_1, dst_2, dst, operator_2.n_components());

#ifdef DEBUG
        VectorType temp;
        temp.reinit(dst);

        VectorTools::merge(this->matrix_free, dst_0, dst_1, dst_2, temp);

        AssertThrow(VectorTools::check_identity(dst, temp), ExcInternalError());
#endif
      }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      AssertDimension(dst.n_blocks(), 1);
      AssertDimension(src.n_blocks(), 1);

      this->vmult(dst.block(0), src.block(0));
    }

    void
    do_update() override
    {
      preconditioner_0->do_update();
      preconditioner_2->do_update();
    }

  private:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    OperatorCahnHilliardHelmholtz<dim, Number, VectorizedArrayType> operator_0;

    MassMatrix<dim, Number, VectorizedArrayType> mass_matrix;

    OperatorAllenCahn<dim, Number, VectorizedArrayType> operator_2;


    const BlockPreconditioner3CHOperator<dim, Number, VectorizedArrayType>
      op_ch;

    mutable VectorType dst_0, dst_1, dst_2;
    mutable VectorType src_0, src_1, src_2;

    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_2;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;

    BlockPreconditioner3CHData data;
  };

} // namespace Sintering
