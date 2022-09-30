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
    {}

    const FreeEnergy<VectorizedArrayType> free_energy;

    const Number kappa_c;
    const Number kappa_p;

    TimeIntegration::TimeIntegratorData<Number> time_data;

  public:
    Table<3, VectorizedArrayType> &
    get_nonlinear_values()
    {
      return nonlinear_values;
    }

    Table<3, VectorizedArrayType> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients()
    {
      return nonlinear_gradients;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
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
    fill_quadrature_point_values(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &src,
      const bool save_op_gradients = false)
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

      nonlinear_values.reinit(
        {n_cells, n_quadrature_points, this->n_components()});

      nonlinear_gradients.reinit({n_cells,
                                  n_quadrature_points,
                                  use_tensorial_mobility || save_op_gradients ?
                                    this->n_components() :
                                    2});

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(matrix_free);

      src.update_ghost_values();

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          phi.reinit(cell);

          for (unsigned int c = 0; c < this->n_components(); ++c)
            {
              phi.read_dof_values_plain(src.block(c));
              phi.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  nonlinear_values(cell, q, c) = phi.get_value(q);

                  if (use_tensorial_mobility || (c < 2) || save_op_gradients)
                    nonlinear_gradients(cell, q, c) = phi.get_gradient(q);
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
      mobility.update(time);
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

    unsigned int number_of_components;

    LinearAlgebra::distributed::DynamicBlockVector<Number> history_vector;
  };
} // namespace Sintering
