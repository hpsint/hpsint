#pragma once

#include <pf-applications/lac/dynamic_block_vector.h>
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
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "cahn_hilliard_op")
      , data(data)
    {}

    unsigned int
    n_components() const override
    {
      return 2;
    }

    unsigned int
    n_grains() const
    {
      return data.n_grains();
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

      const auto &free_energy         = data.free_energy;
      const auto &mobility            = data.mobility;
      const auto &kappa_c             = data.kappa_c;
      const auto &dt                  = data.dt;
      const auto &nonlinear_values    = data.get_nonlinear_values();
      const auto &nonlinear_gradients = data.get_nonlinear_gradients();

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
    const SinteringOperatorData<dim, VectorizedArrayType> &data;
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
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "allen_cahn_op")
      , data(data)
    {}

    unsigned int
    n_components() const override
    {
      return data.n_grains();
    }

    unsigned int
    n_grains() const
    {
      return data.n_grains();
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

      const auto &free_energy      = data.free_energy;
      const auto &L                = data.L;
      const auto &kappa_p          = data.kappa_p;
      const auto &dt               = data.dt;
      const auto &nonlinear_values = data.get_nonlinear_values();

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
    const SinteringOperatorData<dim, VectorizedArrayType> &data;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorAllenCahnBlocked
    : public OperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorAllenCahnBlocked(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const std::string free_energy_approximation_string = "all")
      : OperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "allen_cahn_op")
      , data(data)
      , free_energy_approximation(to_value(free_energy_approximation_string))
      , single_block(free_energy_approximation > 0)
    {}

    static unsigned int
    to_value(const std::string label)
    {
      if (label == "all")
        return 0;
      if (label == "const")
        return 1;
      if (label == "max")
        return 2;
      if (label == "avg")
        return 3;

      AssertThrow(false, ExcNotImplemented());

      return numbers::invalid_unsigned_int;
    }

    unsigned int
    n_components() const override
    {
      return data.n_grains();
    }

    virtual unsigned int
    n_unique_components() const
    {
      return single_block ? 1 : n_components();
    }

    unsigned int
    n_grains() const
    {
      return data.n_grains();
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

      const auto &free_energy      = data.free_energy;
      const auto &L                = data.L;
      const auto &kappa_p          = data.kappa_p;
      const auto &dt               = data.dt;
      const auto &nonlinear_values = data.get_nonlinear_values();

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
            }

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

    const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
    get_block_system_matrix() const
    {
      const bool system_matrix_is_empty = this->block_system_matrix.size() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer,
                        this->label + "::block_matrix::sp",
                        this->do_timing);

          AssertDimension(this->matrix_free.get_dof_handler(this->dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(this->dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);
          dsp.compress();

          this->block_system_matrix.resize(this->n_unique_components());
          for (unsigned int b = 0; b < this->n_unique_components(); ++b)
            {
              this->block_system_matrix[b] =
                std::make_shared<TrilinosWrappers::SparseMatrix>();
              this->block_system_matrix[b]->reinit(dsp);
            }

          this->pcout << std::endl;
          this->pcout << "Create block sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - number of blocks: " << this->n_unique_components()
                      << std::endl;
          this->pcout << " - NNZ:              "
                      << this->block_system_matrix[0]->n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      this->label + "::block_matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          for (unsigned int b = 0; b < this->n_unique_components(); ++b)
            *this->block_system_matrix[b] = 0.0; // clear existing content

        const unsigned int dof_no                   = 0;
        const unsigned int quad_no                  = 0;
        const unsigned int first_selected_component = 0;

        FECellIntegrator<dim, 1, Number, VectorizedArrayType> integrator(
          this->matrix_free, dof_no, quad_no);

        const unsigned int dofs_per_cell = integrator.dofs_per_cell;

        using MatrixType = TrilinosWrappers::SparseMatrix;

        std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
        std::array<std::vector<types::global_dof_index>,
                   VectorizedArrayType::size()>
          dof_indices_mf;
        dof_indices_mf.fill(
          std::vector<types::global_dof_index>(dofs_per_cell));

        std::array<FullMatrix<typename MatrixType::value_type>,
                   VectorizedArrayType::size()>
          matrices;

        std::fill_n(matrices.begin(),
                    VectorizedArrayType::size(),
                    FullMatrix<typename MatrixType::value_type>(dofs_per_cell,
                                                                dofs_per_cell));

        const auto lexicographic_numbering =
          this->matrix_free
            .get_shape_info(dof_no,
                            quad_no,
                            first_selected_component,
                            integrator.get_active_fe_index(),
                            integrator.get_active_quadrature_index())
            .lexicographic_numbering;

        const auto &free_energy      = data.free_energy;
        const auto &L                = data.L;
        const auto &kappa_p          = data.kappa_p;
        const auto &dt               = data.dt;
        const auto &nonlinear_values = data.get_nonlinear_values();

        const auto dt_inv = 1.0 / dt;

        std::vector<const VectorizedArrayType *> etas(this->n_grains());

        for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
             ++cell)
          {
            integrator.reinit(cell);

            const unsigned int n_filled_lanes =
              this->matrix_free.n_active_entries_per_cell_batch(cell);

            // 1) get indices
            for (unsigned int v = 0; v < n_filled_lanes; ++v)
              {
                const auto cell_v =
                  this->matrix_free.get_cell_iterator(cell, v, dof_no);

                if (this->matrix_free.get_mg_level() !=
                    numbers::invalid_unsigned_int)
                  cell_v->get_mg_dof_indices(dof_indices);
                else
                  cell_v->get_dof_indices(dof_indices);

                for (unsigned int j = 0; j < dof_indices.size(); ++j)
                  dof_indices_mf[v][j] =
                    dof_indices[lexicographic_numbering[j]];
              }

            // 2) loop over all blocks
            for (unsigned int b = 0; b < this->n_unique_components(); ++b)
              {
                // 2a) compute columns of blocks
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      integrator.begin_dof_values()[i] =
                        static_cast<Number>(i == j);

                    integrator.evaluate(EvaluationFlags::values |
                                        EvaluationFlags::gradients);

                    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
                      {
                        const auto &val = nonlinear_values[cell][q];
                        const auto &c   = val[0];

                        for (unsigned int ig = 0; ig < this->n_grains(); ++ig)
                          etas[ig] = &val[2 + ig];

                        VectorizedArrayType scaling = 0.0;

                        switch (free_energy_approximation)
                          {
                            case 0:
                              scaling = free_energy.d2f_detai2(c, etas, b);
                              break;
                            case 1:
                              // nothing to do
                              break;
                            case 2:
                              for (unsigned int b = 0; b < this->n_components();
                                   ++b)
                                scaling =
                                  scaling + free_energy.d2f_detai2(c, etas, b);
                              scaling = scaling / static_cast<Number>(
                                                    this->n_components());
                              break;
                            case 3:
                              for (unsigned int b = 0; b < this->n_components();
                                   ++b)
                                for (unsigned int v = 0;
                                     v < VectorizedArrayType::size();
                                     ++v)
                                  {
                                    const auto temp =
                                      free_energy.d2f_detai2(c, etas, b)[v];
                                    scaling[v] =
                                      std::abs(scaling[v]) > std::abs(temp) ?
                                        scaling[v] :
                                        temp;
                                  }
                              break;
                            default:
                              AssertThrow(false, ExcNotImplemented());
                          }

                        const auto value_result =
                          integrator.get_value(q) * dt_inv +
                          L * scaling * integrator.get_value(q);

                        const auto gradient_result =
                          L * kappa_p * integrator.get_gradient(q);

                        integrator.submit_value(value_result, q);
                        integrator.submit_gradient(gradient_result, q);
                      }

                    integrator.integrate(
                      EvaluationFlags::EvaluationFlags::values |
                      EvaluationFlags::EvaluationFlags::gradients);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      for (unsigned int v = 0; v < n_filled_lanes; ++v)
                        matrices[v](i, j) = integrator.begin_dof_values()[i][v];
                  }

                // 2b) compute columns of blocks
                for (unsigned int v = 0; v < n_filled_lanes; ++v)
                  this->constraints.distribute_local_to_global(
                    matrices[v],
                    dof_indices_mf[v],
                    *this->block_system_matrix[b]);
              }
          }
      }

      for (unsigned int b = 0; b < this->n_unique_components(); ++b)
        this->block_system_matrix[b]->compress(VectorOperation::add);

      return this->block_system_matrix;
    }

  private:
    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const unsigned int free_energy_approximation;
    const bool         single_block;
  };


  template <int dim, typename Number, typename VectorizedArrayType>
  class MassMatrix
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          MassMatrix<dim, Number, VectorizedArrayType>>
  {
  public:
    MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
               const AffineConstraints<Number> &                   constraints)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     MassMatrix<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          3,
          "mass_matrix_op")
    {}

    unsigned int
    n_components() const override
    {
      return 1;
    }

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



  struct BlockPreconditioner2Data
  {
    std::string block_0_preconditioner = "ILU";
    std::string block_1_preconditioner = "InverseDiagonalMatrix";

    std::string block_1_approximation = "all";
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
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data,
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const BlockPreconditioner2Data &                       data)
      : data(data)
    {
      // create operators
      operator_0 = std::make_unique<
        OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(matrix_free,
                                                                constraints,
                                                                sintering_data);
      operator_1 =
        std::make_unique<OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
          matrix_free, constraints, sintering_data);
      operator_1_blocked = std::make_unique<
        OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>(
        matrix_free, constraints, sintering_data, data.block_1_approximation);

      // create preconditioners
      preconditioner_0 =
        Preconditioners::create(*operator_0, data.block_0_preconditioner);

      AssertThrow(data.block_1_preconditioner != "BlockGMG",
                  ExcMessage("Use the other constructor!"));

      if (data.block_1_preconditioner == "AMG" ||
          data.block_1_preconditioner == "ILU" ||
          data.block_1_preconditioner == "InverseDiagonalMatrix")
        preconditioner_1 =
          Preconditioners::create(*operator_1, data.block_1_preconditioner);
      else if (data.block_1_preconditioner == "BlockAMG" ||
               data.block_1_preconditioner == "BlockILU")
        preconditioner_1 = Preconditioners::create(*operator_1_blocked,
                                                   data.block_1_preconditioner);
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    BlockPreconditioner2(
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data,
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const MGLevelObject<SinteringOperatorData<dim, VectorizedArrayType>>
        &mg_sintering_data,
      const MGLevelObject<MatrixFree<dim, Number, VectorizedArrayType>>
        &                                             mg_matrix_free,
      const MGLevelObject<AffineConstraints<Number>> &mg_constraints,
      const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
        &                             transfer,
      const BlockPreconditioner2Data &data)
      : data(data)
    {
      const unsigned int min_level = mg_sintering_data.min_level();
      const unsigned int max_level = mg_sintering_data.max_level();

      AssertDimension(min_level, mg_matrix_free.min_level());
      AssertDimension(max_level, mg_matrix_free.max_level());
      AssertDimension(min_level, mg_constraints.min_level());
      AssertDimension(max_level, mg_constraints.max_level());

      // create operators
      operator_0 = std::make_unique<
        OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(matrix_free,
                                                                constraints,
                                                                sintering_data);

      mg_operator_blocked_1.resize(min_level, max_level);
      for (unsigned int l = min_level; l <= max_level; ++l)
        mg_operator_blocked_1[l] = std::make_shared<
          OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>(
          mg_matrix_free[l],
          mg_constraints[l],
          mg_sintering_data[l],
          data.block_1_approximation);

      for (unsigned int l = min_level; l <= max_level; ++l)
        mg_operator_blocked_1[l]->set_timing(false);

      // create preconditioners
      preconditioner_0 =
        Preconditioners::create(*operator_0, data.block_0_preconditioner);

      preconditioner_1 = Preconditioners::create(mg_operator_blocked_1,
                                                 transfer,
                                                 data.block_1_preconditioner);
    }

    virtual void
    clear()
    {
      // clear operators
      if (operator_0)
        operator_0->clear();
      if (operator_1)
        operator_1->clear();
      if (operator_1_blocked)
        operator_1_blocked->clear();

      for (unsigned int l = mg_operator_blocked_1.min_level();
           l <= mg_operator_blocked_1.max_level();
           ++l)
        if (mg_operator_blocked_1[l])
          mg_operator_blocked_1[l]->clear();

      // clear preconditioners
      if (preconditioner_0)
        preconditioner_0->clear();
      if (preconditioner_1)
        preconditioner_1->clear();
    }

    void
    vmult(VectorType &, const VectorType &) const override
    {
      Assert(false, ExcNotImplemented());
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      MyScope scope(timer, "precon::vmult");

      {
        MyScope    scope(timer, "precon::vmult::precon_0");
        const auto dst_view = dst.create_view(0, 2);
        const auto src_view = src.create_view(0, 2);

        preconditioner_0->vmult(*dst_view, *src_view);
      }

      {
        MyScope    scope(timer, "precon::vmult::precon_1");
        const auto dst_view = dst.create_view(2, dst.n_blocks());
        const auto src_view = src.create_view(2, src.n_blocks());

        preconditioner_1->vmult(*dst_view, *src_view);
      }
    }

    void
    do_update() override
    {
      MyScope scope(timer, "precon::update");

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
    // operator CH
    std::unique_ptr<OperatorCahnHilliard<dim, Number, VectorizedArrayType>>
      operator_0;

    // operator AC
    std::unique_ptr<OperatorAllenCahn<dim, Number, VectorizedArrayType>>
      operator_1;
    std::unique_ptr<OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>
      operator_1_blocked;
    MGLevelObject<std::shared_ptr<
      OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>>
      mg_operator_blocked_1;

    // preconditioners
    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_1;

    // utility
    mutable MyTimerOutput timer;

    const BlockPreconditioner2Data data;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class HelmholtzOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          HelmholtzOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    HelmholtzOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const unsigned int                                  n_components_)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     HelmholtzOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "")
      , n_components_(n_components_)
    {}

    unsigned int
    n_components() const override
    {
      return n_components_;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      static_assert(n_grains == -1);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q), q);
        }
    }

  private:
    const unsigned int n_components_;
  };

} // namespace Sintering
