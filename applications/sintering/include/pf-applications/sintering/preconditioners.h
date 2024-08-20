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

#pragma once

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/preconditioners.h>

#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/structural/stvenantkirchhoff.h>

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
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "cahn_hilliard_op")
      , data(data)
      , advection(advection)
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

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      static_assert(n_grains != -1);
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy      = data.free_energy;
      const auto &mobility         = data.get_mobility();
      const auto &kappa_c          = data.kappa_c;
      const auto  weight           = this->data.time_data.get_primary_weight();
      const auto &nonlinear_values = data.get_nonlinear_values();
      const auto &nonlinear_gradients = data.get_nonlinear_gradients();
      const auto  inv_dt = 1. / this->data.time_data.get_current_dt();

      const bool use_coupled_model = data.has_additional_variables_attached();

      // TODO: 1) allow std::array again and 2) allocate less often in the
      // case of std::vector
      std::array<VectorizedArrayType, n_grains>                 etas;
      std::array<Tensor<1, dim, VectorizedArrayType>, n_grains> etas_grad;

      const AdvectionVelocityData<dim, Number, VectorizedArrayType>
        advection_data(cell, this->advection, this->data);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c       = val[0];
          const auto &c_grad  = grad[0];
          const auto &mu_grad = grad[1];

          for (unsigned int ig = 0; ig < etas.size(); ++ig)
            {
              etas[ig] = val[2 + ig];

              if (SinteringOperatorData<dim, VectorizedArrayType>::
                    use_tensorial_mobility)
                etas_grad[ig] = grad[2 + ig];
            }

          typename FECellIntegratorType::value_type    value_result;
          typename FECellIntegratorType::gradient_type gradient_result;

#if true
          // CH with all terms
          value_result[0] = phi.get_value(q)[0] * weight;
          value_result[1] = -phi.get_value(q)[1] +
                            free_energy.d2f_dc2(c, etas) * phi.get_value(q)[0];

          gradient_result[0] =
            mobility.apply_M(
              c, etas, etas.size(), c_grad, etas_grad, phi.get_gradient(q)[1]) +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad *
              phi.get_value(q)[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * phi.get_gradient(q)[0];
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
#else
          // CH with the terms as considered in BlockPreconditioner3CHData
          value_result[0] = phi.get_value(q)[0] * weight;
          value_result[1] = -phi.get_value(q)[1];

          gradient_result[0] = mobility.apply_M(
            c, etas, etas.size(), c_grad, etas_grad, phi.get_gradient(q)[1]);
          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];
#endif

          if (use_coupled_model && this->advection.enabled())
            {
              Tensor<1, dim, VectorizedArrayType> lin_v_adv;
              for (unsigned int d = 0; d < dim; ++d)
                lin_v_adv[d] = val[n_grains + 2 + d] * inv_dt;

              gradient_result[0] -= lin_v_adv * phi.get_value(q)[0];
            }
          else if (this->advection.enabled())
            {
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                if (advection_data.has_velocity(ig))
                  {
                    const auto &velocity_ig =
                      advection_data.get_velocity(ig, phi.quadrature_point(q));

                    gradient_result[0] -= velocity_ig * phi.get_value(q)[0];
                  }
            }

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

  private:
    const SinteringOperatorData<dim, VectorizedArrayType> &     data;
    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class OperatorSolid
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          OperatorSolid<dim, Number, VectorizedArrayType>>
  {
  public:
    OperatorSolid(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const std::array<std::vector<unsigned int>, dim>
        &                                 displ_constraints_indices,
      const double                        E  = 1.0,
      const double                        nu = 0.25,
      const Structural::MaterialPlaneType plane_type =
        Structural::MaterialPlaneType::none)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorSolid<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "solid_op")
      , data(data)
      , displ_constraints_indices(displ_constraints_indices)
      , material(E, nu, plane_type)
    {}

    unsigned int
    n_components() const override
    {
      return dim;
    }

    unsigned int
    n_grains() const
    {
      return data.n_grains();
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int)
    {
      return dim;
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      static_assert(n_grains != -1);
      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values = data.get_nonlinear_values();

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val = nonlinear_values[cell][q];

          const auto &c = val[0];

          const auto S = this->get_stress(phi.get_gradient(q), c);

          phi.submit_value({}, q);
          phi.submit_gradient(S, q);
        }
    }

    void
    post_system_matrix_compute() const override
    {
      const auto &partitioner = this->matrix_free.get_vector_partitioner();

      for (unsigned int d = 0; d < dim; ++d)
        for (const unsigned int index : displ_constraints_indices[d])
          {
            const auto global_index = partitioner->local_to_global(index);

            const unsigned int matrix_index = dim * global_index + d;

            this->system_matrix.clear_row(matrix_index, 1.0);
          }
    }

    Tensor<2, dim, VectorizedArrayType>
    get_stress(const Tensor<2, dim, VectorizedArrayType> &H,
               const VectorizedArrayType &                c) const
    {
      const double c_min = 0.1;

      const auto cl = compare_and_apply_mask<SIMDComparison::less_than>(
        c, VectorizedArrayType(c_min), VectorizedArrayType(c_min), c);

      return cl * material.get_S(H);
    }

  private:
    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const std::array<std::vector<unsigned int>, dim> &displ_constraints_indices;

    const Structural::StVenantKirchhoff<dim, Number, VectorizedArrayType>
      material;
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
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "allen_cahn_op")
      , data(data)
      , advection(advection)
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

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      static_assert(n_comp == n_grains);

      if constexpr (n_comp > 1)
        {
          const unsigned int cell = phi.get_current_cell_index();

          const auto &free_energy = data.free_energy;
          const auto &L           = data.get_mobility().Lgb();
          const auto &kappa_p     = data.kappa_p;
          const auto  weight      = this->data.time_data.get_primary_weight();
          const auto &nonlinear_values = data.get_nonlinear_values();
          const auto  inv_dt = 1. / this->data.time_data.get_current_dt();

          const bool use_coupled_model =
            data.has_additional_variables_attached();

          const AdvectionVelocityData<dim, Number, VectorizedArrayType>
            advection_data(cell, this->advection, this->data);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto &val = nonlinear_values[cell][q];

              const auto &c = val[0];

              std::array<VectorizedArrayType, n_grains> etas;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                etas[ig] = val[2 + ig];

              typename FECellIntegratorType::value_type    value_result;
              typename FECellIntegratorType::gradient_type gradient_result;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[ig] = phi.get_value(q)[ig] * weight +
                                     L * free_energy.d2f_detai2(c, etas, ig) *
                                       phi.get_value(q)[ig];

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

              if (use_coupled_model && this->advection.enabled())
                {
                  Tensor<1, dim, VectorizedArrayType> lin_v_adv;
                  for (unsigned int d = 0; d < dim; ++d)
                    lin_v_adv[d] = val[n_grains + 2 + d] * inv_dt;

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    gradient_result[ig] -= lin_v_adv * phi.get_value(q)[ig];
                }
              else if (this->advection.enabled())
                {
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    if (advection_data.has_velocity(ig))
                      {
                        const auto &velocity_ig =
                          advection_data.get_velocity(ig,
                                                      phi.quadrature_point(q));

                        gradient_result[ig] -=
                          velocity_ig * phi.get_value(q)[ig];
                      }
                }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          (void)phi;
        }
    }

  private:
    const SinteringOperatorData<dim, VectorizedArrayType> &     data;
    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;
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
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
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
      , advection(advection)
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
    n_unique_components() const override
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

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      static_assert(n_comp == n_grains);

      if constexpr (n_comp > 1)
        {
          const unsigned int cell = phi.get_current_cell_index();

          const auto &free_energy      = data.free_energy;
          const auto &L                = data.get_mobility().Lgb();
          const auto &kappa_p          = data.kappa_p;
          const auto  weight           = data.time_data.get_primary_weight();
          const auto &nonlinear_values = data.get_nonlinear_values();
          const auto  inv_dt = 1. / this->data.time_data.get_current_dt();

          const bool use_coupled_model =
            data.has_additional_variables_attached();

          const AdvectionVelocityData<dim, Number, VectorizedArrayType>
            advection_data(cell, this->advection, this->data);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto &val = nonlinear_values[cell][q];

              const auto &c = val[0];

              std::array<VectorizedArrayType, n_grains> etas;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                etas[ig] = val[2 + ig];

              typename FECellIntegratorType::value_type    value_result;
              typename FECellIntegratorType::gradient_type gradient_result;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[ig] = phi.get_value(q)[ig] * weight +
                                     L * free_energy.d2f_detai2(c, etas, ig) *
                                       phi.get_value(q)[ig];

                  gradient_result[ig] = L * kappa_p * phi.get_gradient(q)[ig];
                }

              if (use_coupled_model && this->advection.enabled())
                {
                  Tensor<1, dim, VectorizedArrayType> lin_v_adv;
                  for (unsigned int d = 0; d < dim; ++d)
                    lin_v_adv[d] = val[n_grains + 2 + d] * inv_dt;

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    gradient_result[ig] -= lin_v_adv * phi.get_value(q)[ig];
                }
              else if (this->advection.enabled())
                {
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    if (advection_data.has_velocity(ig))
                      {
                        const auto &velocity_ig =
                          advection_data.get_velocity(ig,
                                                      phi.quadrature_point(q));

                        gradient_result[ig] -=
                          velocity_ig * phi.get_value(q)[ig];
                      }
                }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          (void)phi;
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

          TrilinosWrappers::SparsityPattern dsp;
          dsp.reinit(dof_handler.locally_owned_dofs(),
                     dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler),
                     dof_handler.get_communicator());

          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          this->constraints,
                                          this->matrix_free.get_quadrature());
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
        const auto &L                = data.get_mobility().Lgb();
        const auto &kappa_p          = data.kappa_p;
        const auto  weight           = data.time_data.get_primary_weight();
        const auto &nonlinear_values = data.get_nonlinear_values();
        const auto  inv_dt = 1. / this->data.time_data.get_current_dt();

        const bool use_coupled_model = data.has_additional_variables_attached();

        const auto &component_table = this->data.get_component_table();

        std::vector<VectorizedArrayType> etas(this->n_grains());

        AdvectionVelocityData<dim, Number, VectorizedArrayType> advection_data(
          this->advection, this->data);

        for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
             ++cell)
          {
            integrator.reinit(cell);

            const unsigned int n_filled_lanes =
              this->matrix_free.n_active_entries_per_cell_batch(cell);

            advection_data.reinit(cell);

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

            AlignedVector<VectorizedArrayType> scaling(integrator.n_q_points);

            for (unsigned int q = 0; q < integrator.n_q_points; ++q)
              {
                const auto &val = nonlinear_values[cell][q];
                const auto &c   = val[0];

                for (unsigned int ig = 0; ig < this->n_grains(); ++ig)
                  etas[ig] = val[2 + ig];

                switch (free_energy_approximation)
                  {
                    case 0:
                      // nothing to do -> done later
                      break;
                    case 1:
                      // nothing to do
                      break;
                    case 2:
                      for (unsigned int b = 0; b < this->n_components(); ++b)
                        scaling[q] =
                          scaling[q] + free_energy.d2f_detai2(c, etas, b);
                      scaling[q] =
                        scaling[q] / static_cast<Number>(this->n_components());
                      break;
                    case 3:
                      for (unsigned int b = 0; b < this->n_components(); ++b)
                        for (unsigned int v = 0;
                             v < VectorizedArrayType::size();
                             ++v)
                          {
                            const auto temp =
                              free_energy.d2f_detai2(c, etas, b)[v];
                            scaling[q][v] =
                              std::abs(scaling[q][v]) > std::abs(temp) ?
                                scaling[q][v] :
                                temp;
                          }
                      break;
                    default:
                      AssertThrow(false, ExcNotImplemented());
                  }
              }

            // 2) loop over all blocks
            for (unsigned int b = 0; b < this->n_unique_components(); ++b)
              {
                if (free_energy_approximation == 0)
                  {
                    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
                      {
                        const auto &val = nonlinear_values[cell][q];
                        const auto &c   = val[0];

                        for (unsigned int ig = 0; ig < this->n_grains(); ++ig)
                          etas[ig] = val[2 + ig];
                        scaling[q] = free_energy.d2f_detai2(c, etas, b);
                      }
                  }

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

                        auto value    = integrator.get_value(q);
                        auto gradient = integrator.get_gradient(q);

                        if (free_energy_approximation == 0 &&
                            component_table.size(0) > 0)
                          if (component_table[cell][b] == false)
                            {
                              value    = VectorizedArrayType();
                              gradient = Tensor<1, dim, VectorizedArrayType>();
                            }

                        auto value_result =
                          value * weight + L * scaling[q] * value;
                        auto gradient_result = L * kappa_p * gradient;

                        if (free_energy_approximation == 0 &&
                            this->advection.enabled())
                          {
                            if (use_coupled_model)
                              {
                                Tensor<1, dim, VectorizedArrayType> lin_v_adv;
                                for (unsigned int d = 0; d < dim; ++d)
                                  lin_v_adv[d] =
                                    val[this->n_grains() + 2 + d] * inv_dt;

                                gradient_result -= lin_v_adv * value;
                              }
                            else if (advection_data.has_velocity(b))
                              {
                                const auto &velocity_ig =
                                  advection_data.get_velocity(
                                    b, integrator.quadrature_point(q));

                                gradient_result -= velocity_ig * value;
                              }
                          }


                        if (free_energy_approximation == 0 &&
                            component_table.size(0) > 0)
                          if (component_table[cell][b] == false)
                            {
                              value_result = VectorizedArrayType();
                              gradient_result =
                                Tensor<1, dim, VectorizedArrayType>();
                            }

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
                  {
                    // new: remove small entries (TODO: only for FE_Q_iso_1)
                    Number max = 0.0;

                    for (unsigned int i = 0; i < matrices[v].m(); ++i)
                      for (unsigned int j = 0; j < matrices[v].n(); ++j)
                        max = std::max(max, std::abs(matrices[v][i][j]));

                    for (unsigned int i = 0; i < matrices[v].m(); ++i)
                      for (unsigned int j = 0; j < matrices[v].n(); ++j)
                        if (std::abs(matrices[v][i][j]) < 1e-10 * max)
                          matrices[v][i][j] = 0.0;

                    this->constraints.distribute_local_to_global(
                      matrices[v],
                      dof_indices_mf[v],
                      *this->block_system_matrix[b]);
                  }
              }
          }
      }

      for (unsigned int b = 0; b < this->n_unique_components(); ++b)
        {
          auto &matrix = *this->block_system_matrix[b];

          matrix.compress(VectorOperation::add);

          const auto &component_table = this->data.get_component_table();

          if (free_energy_approximation == 0 && component_table.size(0) > 0)
            {
              const auto range = matrix.local_range();

              for (unsigned int r = range.first; r < range.second; ++r)
                {
                  auto entry = matrix.begin(r);
                  auto end   = matrix.end(r);

                  for (; entry != end; ++entry)
                    if (entry->row() == entry->column() &&
                        entry->value() == 0.0)
                      entry->value() = 1.0;
                }

              matrix.compress(VectorOperation::insert);
            }
        }

      return this->block_system_matrix;
    }

  private:
    const SinteringOperatorData<dim, VectorizedArrayType> &     data;
    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;

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
          0,
          "mass_matrix_op")
    {}

    unsigned int
    n_components() const override
    {
      return 1;
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      static_assert(n_grains == -1);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(typename FECellIntegratorType::gradient_type(),
                              q);
        }
    }
  };

  struct AMGData
  {
    unsigned int smoother_sweeps = 4;
    unsigned int n_cycles        = 5;
  };

  struct BlockPreconditioner2Data
  {
    std::string block_0_preconditioner = "ILU";
    std::string block_1_preconditioner = "InverseDiagonalMatrix";
    std::string block_2_preconditioner = "AMG";

    std::string block_1_approximation = "all";

    AMGData block_2_amg_data;
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
      const BlockPreconditioner2Data &                       data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection)
      : data(data)
    {
      // create operators
      operator_0 = std::make_unique<
        OperatorCahnHilliard<dim, Number, VectorizedArrayType>>(matrix_free,
                                                                constraints,
                                                                sintering_data,
                                                                advection);
      operator_1 =
        std::make_unique<OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
          matrix_free, constraints, sintering_data, advection);
      operator_1_blocked = std::make_unique<
        OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>(
        matrix_free,
        constraints,
        sintering_data,
        advection,
        data.block_1_approximation);

      // create preconditioners
      preconditioner_0 =
        Preconditioners::create(*operator_0, data.block_0_preconditioner);

      AssertThrow((data.block_1_preconditioner != "GMG") &&
                    (data.block_1_preconditioner != "BlockGMG"),
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
      const BlockPreconditioner2Data &                       data,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const std::array<std::vector<unsigned int>, dim>
        &                                 zero_constraints_indices,
      const double                        E  = 1.0,
      const double                        nu = 0.25,
      const Structural::MaterialPlaneType plane_type =
        Structural::MaterialPlaneType::none)
      : BlockPreconditioner2(sintering_data,
                             matrix_free,
                             constraints,
                             data,
                             advection)
    {
      operator_2 =
        std::make_unique<OperatorSolid<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          sintering_data,
          zero_constraints_indices,
          E,
          nu,
          plane_type);

      if (data.block_2_preconditioner == "AMG")
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;
          additional_data.smoother_sweeps =
            data.block_2_amg_data.smoother_sweeps;
          additional_data.n_cycles = data.block_2_amg_data.n_cycles;
          preconditioner_2 =
            Preconditioners::create(*operator_2,
                                    data.block_2_preconditioner,
                                    additional_data);
        }
      else
        {
          preconditioner_2 =
            Preconditioners::create(*operator_2, data.block_2_preconditioner);
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
      AssertThrow(false, ExcNotImplemented());

      AdvectionMechanism<dim, Number, VectorizedArrayType> advection;

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
                                                                sintering_data,
                                                                advection);

      if (data.block_1_preconditioner == "GMG")
        {
          mg_operator_1.resize(min_level, max_level);
          for (unsigned int l = min_level; l <= max_level; ++l)
            mg_operator_1[l] = std::make_shared<
              OperatorAllenCahn<dim, Number, VectorizedArrayType>>(
              mg_matrix_free[l],
              mg_constraints[l],
              mg_sintering_data[l],
              advection);
          for (unsigned int l = min_level; l <= max_level; ++l)
            mg_operator_1[l]->set_timing(false);
        }
      else
        {
          mg_operator_blocked_1.resize(min_level, max_level);
          for (unsigned int l = min_level; l <= max_level; ++l)
            mg_operator_blocked_1[l] = std::make_shared<
              OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>(
              mg_matrix_free[l],
              mg_constraints[l],
              mg_sintering_data[l],
              advection,
              data.block_1_approximation);
          for (unsigned int l = min_level; l <= max_level; ++l)
            mg_operator_blocked_1[l]->set_timing(false);
        }

      // create preconditioners
      preconditioner_0 =
        Preconditioners::create(*operator_0, data.block_0_preconditioner);

      if (data.block_1_preconditioner == "GMG")
        preconditioner_1 = Preconditioners::create(mg_operator_1,
                                                   transfer,
                                                   data.block_1_preconditioner);
      else
        preconditioner_1 = Preconditioners::create(mg_operator_blocked_1,
                                                   transfer,
                                                   data.block_1_preconditioner);
    }

    virtual void
    clear() override
    {
      // clear operators
      if (operator_0)
        operator_0->clear();
      if (operator_1)
        operator_1->clear();
      if (operator_1_blocked)
        operator_1_blocked->clear();
      if (operator_2)
        operator_2->clear();

      for (unsigned int l = mg_operator_1.min_level();
           l <= mg_operator_1.max_level();
           ++l)
        if (mg_operator_1[l])
          mg_operator_1[l]->clear();

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
      if (preconditioner_2)
        preconditioner_2->clear();
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
        const auto start    = 0;
        const auto end      = 2;
        const auto dst_view = dst.create_view(start, end);
        const auto src_view = src.create_view(start, end);

        preconditioner_0->vmult(*dst_view, *src_view);
      }

      {
        MyScope    scope(timer, "precon::vmult::precon_1");
        const auto start = 2;
        const auto end =
          preconditioner_2 ? (dst.n_blocks() - dim) : dst.n_blocks();
        const auto dst_view = dst.create_view(start, end);
        const auto src_view = src.create_view(start, end);

        preconditioner_1->vmult(*dst_view, *src_view);
      }

      if (preconditioner_2)
        {
          MyScope    scope(timer, "precon::vmult::precon_2");
          const auto start    = dst.n_blocks() - dim;
          const auto end      = dst.n_blocks();
          const auto dst_view = dst.create_view(start, end);
          const auto src_view = src.create_view(start, end);

          preconditioner_2->vmult(*dst_view, *src_view);
        }
    }

    void
    do_update() override
    {
      MyScope scope(timer, "precon::update");

      if (preconditioner_0)
        {
          MyScope scope(timer, "precon::update::precon_0");
          preconditioner_0->do_update();
        }
      if (preconditioner_1)
        {
          MyScope scope(timer, "precon::update::precon_1");
          preconditioner_1->do_update();
        }
      if (preconditioner_2)
        {
          MyScope scope(timer, "precon::update::precon_2");
          preconditioner_2->do_update();
        }
    }

    virtual std::size_t
    memory_consumption() const override
    {
      return MyMemoryConsumption::memory_consumption(operator_0) +
             MyMemoryConsumption::memory_consumption(operator_1) +
             MyMemoryConsumption::memory_consumption(operator_1_blocked) +
             MyMemoryConsumption::memory_consumption(mg_operator_1) +
             MyMemoryConsumption::memory_consumption(mg_operator_blocked_1) +
             MyMemoryConsumption::memory_consumption(preconditioner_0) +
             MyMemoryConsumption::memory_consumption(preconditioner_1);
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

    // operator solid
    std::unique_ptr<OperatorSolid<dim, Number, VectorizedArrayType>> operator_2;

    MGLevelObject<
      std::shared_ptr<OperatorAllenCahn<dim, Number, VectorizedArrayType>>>
      mg_operator_1;
    MGLevelObject<std::shared_ptr<
      OperatorAllenCahnBlocked<dim, Number, VectorizedArrayType>>>
      mg_operator_blocked_1;

    // preconditioners
    std::unique_ptr<Preconditioners::PreconditionerBase<Number>>
      preconditioner_0, preconditioner_1, preconditioner_2;

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

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
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
