// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the hpsint authors
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

#include <pf-applications/sintering/mobility.h>

#include <pf-applications/time_integration/time_integrators.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename VectorizedArrayType>
  struct SinteringOperatorData
  {
    using Number = typename VectorizedArrayType::value_type;

    // Choose MobilityScalar or MobilityTensorial here:
    static const bool use_tensorial_mobility =
#ifdef WITH_TENSORIAL_MOBILITY
      true;
#else
      false;
#endif

    using MobilityType =
      typename std::conditional<use_tensorial_mobility,
                                MobilityTensorial<dim, VectorizedArrayType>,
                                MobilityScalar<dim, VectorizedArrayType>>::type;

    SinteringOperatorData(const Number                      kappa_c,
                          const Number                      kappa_p,
                          std::shared_ptr<MobilityProvider> mobility_provider,
                          const unsigned int                integration_order,
                          const unsigned int op_components_offset = 2)
      : kappa_c(kappa_c)
      , kappa_p(kappa_p)
      , time_data(integration_order)
      , mobility(mobility_provider)
      , op_components_offset(op_components_offset)
      , t(0.0)
    {}

    const Number kappa_c;
    const Number kappa_p;

    TimeIntegration::TimeIntegratorData<Number> time_data;

  public:
    const Table<3, VectorizedArrayType> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }

    const Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
    }

    const VectorizedArrayType *
    get_nonlinear_values(const unsigned int cell) const
    {
      if (value_ptr.empty())
        return &nonlinear_values[cell][0][0];
      else
        return &nonlinear_values_new[value_ptr[cell]];
    }

    const dealii::Tensor<1, dim, VectorizedArrayType> *
    get_nonlinear_gradients(const unsigned int cell) const
    {
      if (gradient_ptr.empty())
        return &nonlinear_gradients[cell][0][0];
      else
        return &nonlinear_gradients_new[gradient_ptr[cell]];
    }

    Table<2, bool> &
    get_component_table() const
    {
      return component_table;
    }

    void
    set_n_components(const unsigned int number_of_components)
    {
      AssertThrow(
        number_of_components >= op_components_offset,
        ExcMessage(
          "The new total number of components (" +
          std::to_string(number_of_components) +
          ") should not be smaller than the number of CH components (" +
          std::to_string(op_components_offset) + ")."));

      this->number_of_components = number_of_components;
    }

    unsigned int
    n_components() const
    {
      return number_of_components;
    }

    unsigned int
    n_grains() const
    {
      return number_of_components - op_components_offset;
    }

    unsigned int
    n_non_grains() const
    {
      return op_components_offset;
    }

    ComponentMask
    build_pf_component_mask(const unsigned int n_total_components) const
    {
      ComponentMask mask(n_total_components, false);
      mask.set(0, true); // CH concentration, CH chemical potential is skipped
      for (unsigned int g = 0; g < n_grains(); ++g)
        mask.set(op_components_offset + g, true); // AC order parameters

      return mask;
    }

    void
    set_component_mask(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &src,
      const bool   save_op_gradients,
      const bool   save_all_blocks,
      const double grain_use_cut_off_tolerance)
    {
      src.update_ghost_values();

      const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

      const auto comm = MPI_COMM_WORLD;

      const unsigned n_cells = matrix_free.n_cell_batches();

      Table<2, unsigned int> table_lanes;
      table_lanes.reinit({n_cells, VectorizedArrayType::size()});

      for (unsigned int i = 0; i < n_cells; ++i)
        for (unsigned int j = 0; j < n_grains(); ++j)
          table_lanes[i][j] = numbers::invalid_unsigned_int;

      component_table.reinit({n_cells, n_grains()});

      for (unsigned int i = 0; i < n_cells; ++i)
        for (unsigned int j = 0; j < n_grains(); ++j)
          component_table[i][j] = false;

      const auto &dof_handler = matrix_free.get_dof_handler();

      Vector<Number> values(dof_handler.get_fe().n_dofs_per_cell());

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_cell_batch(cell);
               ++v)
            {
              const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

              unsigned int counter = 0;

              for (unsigned int b = 0; b < n_grains(); ++b)
                {
                  cell_iterator->get_dof_values(src.block(b +
                                                          op_components_offset),
                                                values);

                  if (values.linfty_norm() > grain_use_cut_off_tolerance)
                    {
                      component_table[cell][b] = true;
                      counter++;
                    }
                }

              table_lanes[cell][v] = counter;
            }
        }

      value_ptr    = {0};
      gradient_ptr = {0};

      relevant_grains_vector = {};
      relevant_grains_ptr    = {0};

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          unsigned int counter = 0;

          for (unsigned int j = 0; j < n_grains(); ++j)
            if (component_table[cell][j])
              {
                counter++;
                relevant_grains_vector.push_back(j);
              }

          relevant_grains_ptr.push_back(relevant_grains_vector.size());

          const unsigned n_components_save_value =
            (save_all_blocks ? src.n_blocks() : this->n_components()) -
            n_grains() + counter;

          const unsigned n_components_save_gradient =
            use_tensorial_mobility || save_op_gradients ?
              n_components_save_value :
              op_components_offset;

          value_ptr.emplace_back(value_ptr.back() +
                                 n_quadrature_points * n_components_save_value);
          gradient_ptr.emplace_back(gradient_ptr.back() +
                                    n_quadrature_points *
                                      n_components_save_gradient);
        }

      ConditionalOStream pcout(std::cout,
                               Utilities::MPI::this_mpi_process(comm) == 0);

      // some statistics
      const auto print_stat = [&pcout,
                               &comm](std::vector<unsigned int> &counters) {
        unsigned int max_value =
          *std::max_element(counters.begin(), counters.end());
        max_value = Utilities::MPI::max(max_value, comm);

        std::vector<unsigned int> max_values(max_value + 1, 0);

        for (const auto i : counters)
          max_values[i]++;

        Utilities::MPI::sum(max_values, comm, max_values);

        const auto sum = std::reduce(max_values.begin(), max_values.end());

        pcout << "  - " << max_value << " (";

        pcout << (0) << ": " << static_cast<double>(max_values[0]) / sum * 100;
        for (unsigned int i = 1; i < max_values.size(); ++i)
          pcout << ", " << i << ": "
                << static_cast<double>(max_values[i]) / sum * 100;

        pcout << ")" << std::endl;
      };

      std::vector<unsigned int> counters_batch_max(n_cells, 0);
      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int j = 0; j < n_grains(); ++j)
            if (component_table[i][j])
              counters_batch_max[i]++;
        }

      std::vector<unsigned int> counters_batch_compressed(n_cells, 0);
      std::vector<unsigned int> counters_cell;
      for (unsigned int i = 0; i < n_cells; ++i)
        for (unsigned int j = 0; j < VectorizedArrayType::size(); ++j)
          if (table_lanes[i][j] != numbers::invalid_unsigned_int)
            {
              counters_batch_compressed[i] =
                std::max(counters_batch_compressed[i], table_lanes[i][j]);
              counters_cell.push_back(table_lanes[i][j]);
            }

      pcout << "Cut-off statistic: " << std::endl;
      print_stat(counters_batch_max);
      print_stat(counters_batch_compressed);
      print_stat(counters_cell);

      src.zero_out_ghost_values();
    }

    void
    fill_quadrature_point_values(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &src,
      const bool save_op_gradients = false,
      const bool save_all_blocks   = false)
    {
      AssertThrow(src.n_blocks() >= this->n_components(),
                  ExcMessage("Source vector size (" +
                             std::to_string(src.n_blocks()) +
                             ") is too small to fill all the components (" +
                             std::to_string(this->n_components()) + ")."));

      this->history_vector = src;
      this->history_vector.update_ghost_values();

      const unsigned n_cells             = matrix_free.n_cell_batches();
      const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

      const unsigned n_components_save =
        save_all_blocks ? src.n_blocks() : this->n_components();

      nonlinear_values.reinit(
        {n_cells, n_quadrature_points, n_components_save});

      nonlinear_gradients.reinit({n_cells,
                                  n_quadrature_points,
                                  use_tensorial_mobility || save_op_gradients ?
                                    n_components_save :
                                    op_components_offset});

      if (value_ptr.empty() == false)
        nonlinear_values_new.resize(value_ptr.back());

      if (gradient_ptr.empty() == false)
        nonlinear_gradients_new.resize(gradient_ptr.back());

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(matrix_free);

      src.update_ghost_values();

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          phi.reinit(cell);

          for (unsigned int c = 0; c < n_components_save; ++c)
            {
              phi.read_dof_values_plain(src.block(c));
              phi.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  auto value    = phi.get_value(q);
                  auto gradient = phi.get_gradient(q);

                  if ((component_table.size(0) > 0) &&
                      (c >= op_components_offset))
                    if (component_table[cell][c - op_components_offset] ==
                        false)
                      {
                        value    = VectorizedArrayType();
                        gradient = Tensor<1, dim, VectorizedArrayType>();
                      }

                  nonlinear_values(cell, q, c) = value;

                  if (use_tensorial_mobility || (c < op_components_offset) ||
                      save_op_gradients)
                    nonlinear_gradients(cell, q, c) = gradient;
                }
            }

          if (component_table.size(0) > 0)
            {
              const unsigned n_components_save_value =
                save_all_blocks ? src.n_blocks() : this->n_components();

              const unsigned n_components_save_gradient =
                use_tensorial_mobility || save_op_gradients ?
                  n_components_save_value :
                  op_components_offset;

              unsigned int counter_v = value_ptr[cell];
              unsigned int counter_g = gradient_ptr[cell];

              for (unsigned int q = 0; q < n_quadrature_points; ++q)
                {
                  for (unsigned int c = 0; c < n_components_save_value; ++c)
                    if (((c < op_components_offset) ||
                         (c >= (op_components_offset + n_grains()))) ||
                        component_table[cell][c - op_components_offset])
                      nonlinear_values_new[counter_v++] =
                        nonlinear_values(cell, q, c);

                  for (unsigned int c = 0; c < n_components_save_gradient; ++c)
                    if (((c < op_components_offset) ||
                         (c >= (op_components_offset + n_grains()))) ||
                        component_table[cell][c - op_components_offset])
                      nonlinear_gradients_new[counter_g++] =
                        nonlinear_gradients(cell, q, c);
                }

              AssertDimension(counter_v, value_ptr[cell + 1]);
              AssertDimension(counter_g, gradient_ptr[cell + 1]);
            }
        }

      src.zero_out_ghost_values();
    }

    virtual std::size_t
    memory_consumption() const
    {
      return nonlinear_values.memory_consumption() +
             nonlinear_gradients.memory_consumption();
    }

    const LinearAlgebra::distributed::DynamicBlockVector<Number> &
    get_history_vector() const
    {
      return history_vector;
    }

    void
    set_time(const double time)
    {
      t = time;
      mobility.update(time);
    }

    double
    get_time() const
    {
      return t;
    }

    const MobilityType &
    get_mobility() const
    {
      return mobility;
    }

    ArrayView<const unsigned char>
    get_relevant_grains(const unsigned int cell) const
    {
      return ArrayView<const unsigned char>(relevant_grains_vector.data() +
                                              relevant_grains_ptr[cell],
                                            relevant_grains_ptr[cell + 1] -
                                              relevant_grains_ptr[cell]);
    }

    std::vector<unsigned char>
    get_grain_to_relevant_grain(const unsigned cell) const
    {
      std::vector<unsigned char> result(this->n_grains(),
                                        static_cast<unsigned char>(255));

      if (!cut_off_enabled())
        {
          for (unsigned int i = 0; i < this->n_grains(); ++i)
            result[i] = i;

          return result;
        }

      const auto relevant_grains = get_relevant_grains(cell);

      for (unsigned int i = 0; i < relevant_grains.size(); ++i)
        result[relevant_grains[i]] = i;

      return result;
    }

    bool
    cut_off_enabled() const
    {
      return !relevant_grains_ptr.empty();
    }

  private:
    MobilityType mobility;

    const unsigned int op_components_offset;

    mutable Table<3, VectorizedArrayType> nonlinear_values;
    mutable Table<3, dealii::Tensor<1, dim, VectorizedArrayType>>
      nonlinear_gradients;

    mutable AlignedVector<VectorizedArrayType> nonlinear_values_new;
    mutable AlignedVector<dealii::Tensor<1, dim, VectorizedArrayType>>
      nonlinear_gradients_new;

    std::vector<unsigned int> value_ptr;
    std::vector<unsigned int> gradient_ptr;

    std::vector<unsigned char> relevant_grains_vector;
    std::vector<unsigned int>  relevant_grains_ptr;

    mutable Table<2, bool> component_table;

    unsigned int number_of_components;

    LinearAlgebra::distributed::DynamicBlockVector<Number> history_vector;

    double t;
  };

  template <int dim, typename VectorizedArrayType>
  struct SinteringNonLinearData
  {
    decltype(std::declval<const Table<3, VectorizedArrayType>>().operator[](0))
      const &values;
    decltype(
      std::declval<const Table<3, Tensor<1, dim, VectorizedArrayType>>>().
      operator[](0)) const &gradients;
  };
} // namespace Sintering
