#pragma once

#include <pf-applications/sintering/free_energy.h>
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

    SinteringOperatorData(const Number                      A,
                          const Number                      B,
                          const Number                      kappa_c,
                          const Number                      kappa_p,
                          std::shared_ptr<MobilityProvider> mobility_provider,
                          const unsigned int                integration_order)
      : free_energy(A, B)
      , kappa_c(kappa_c)
      , kappa_p(kappa_p)
      , time_data(integration_order)
      , mobility(mobility_provider)
      , t(0.0)
    {}

    const FreeEnergy<VectorizedArrayType> free_energy;

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
      return &nonlinear_values[cell][0][0];
    }

    const dealii::Tensor<1, dim, VectorizedArrayType> *
    get_nonlinear_gradients(const unsigned int cell) const
    {
      return &nonlinear_gradients[cell][0][0];
    }

    Table<2, bool> &
    get_component_table() const
    {
      return component_table;
    }

    void
    set_n_components(const unsigned int number_of_components)
    {
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
      return number_of_components - 2;
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

              for (unsigned int b = 0; b < n_grains(); ++b)
                {
                  cell_iterator->get_dof_values(src.block(b + 2), values);

                  if (values.linfty_norm() > grain_use_cut_off_tolerance)
                    component_table[cell][b] = true;
                }
            }
        }

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          unsigned int counter = 0;

          for (unsigned int j = 0; j < n_grains(); ++j)
            if (component_table[cell][j])
              counter++;

          const unsigned n_components_save_value =
            (save_all_blocks ? src.n_blocks() : this->n_components()) -
            n_grains() + counter;

          const unsigned n_components_save_gradient =
            use_tensorial_mobility || save_op_gradients ?
              n_components_save_value :
              2;

          value_ptr.emplace_back(value_ptr.back() +
                                 n_quadrature_points * n_components_save_value);
          gradient_ptr.emplace_back(gradient_ptr.back() +
                                    n_quadrature_points *
                                      n_components_save_gradient);
        }


      // some statistics
      std::vector<unsigned int> counters(n_cells, 0);

      for (unsigned int i = 0; i < n_cells; ++i)
        {
          for (unsigned int j = 0; j < n_grains(); ++j)
            if (component_table[i][j])
              counters[i]++;
        }

      unsigned int max_value =
        *std::max_element(counters.begin(), counters.end());
      max_value = Utilities::MPI::max(max_value, comm);

      std::vector<unsigned int> max_values(max_value, 0);

      for (const auto i : counters)
        if (i != 0)
          max_values[i - 1]++;

      Utilities::MPI::sum(max_values, comm, max_values);

      ConditionalOStream pcout(std::cout,
                               Utilities::MPI::this_mpi_process(comm) == 0);

      pcout << "Cut-off statistic: " << max_value << " (";

      pcout << (1) << ": " << max_values[0];
      for (unsigned int i = 1; i < max_values.size(); ++i)
        pcout << ", " << (i + 1) << ": " << max_values[i];

      pcout << ")" << std::endl;

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

      nonlinear_gradients.reinit(
        {n_cells,
         n_quadrature_points,
         use_tensorial_mobility || save_op_gradients ? n_components_save : 2});

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

                  if ((component_table.size(0) > 0) && (c >= 2))
                    if (component_table[cell][c - 2] == false)
                      {
                        value    = VectorizedArrayType();
                        gradient = Tensor<1, dim, VectorizedArrayType>();
                      }

                  nonlinear_values(cell, q, c) = value;

                  if (use_tensorial_mobility || (c < 2) || save_op_gradients)
                    nonlinear_gradients(cell, q, c) = gradient;
                }
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

  private:
    MobilityType mobility;

    mutable Table<3, VectorizedArrayType> nonlinear_values;
    mutable Table<3, dealii::Tensor<1, dim, VectorizedArrayType>>
      nonlinear_gradients;


    std::vector<unsigned int> value_ptr;
    std::vector<unsigned int> gradient_ptr;

    mutable Table<2, bool> component_table;

    unsigned int number_of_components;

    LinearAlgebra::distributed::DynamicBlockVector<Number> history_vector;

    double t;
  };
} // namespace Sintering
