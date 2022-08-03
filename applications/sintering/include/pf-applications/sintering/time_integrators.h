#include <pf-applications/base/fe_integrator.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <array>

namespace Sintering
{
  namespace internal
  {
    template <typename T, std::size_t... Is>
    constexpr std::array<T, sizeof...(Is)>
    create_array(T value, std::index_sequence<Is...>)
    {
      // cast Is to void to remove the warning: unused value
      return {{(static_cast<void>(Is), value)...}};
    }
  } // namespace internal

  template <std::size_t N, typename T>
  constexpr std::array<T, N>
  create_array(const T &value)
  {
    return internal::create_array(value, std::make_index_sequence<N>());
  }

  using namespace dealii;

  template <typename Number, int order>
  class TimeIntegratorData
  {
  public:
    TimeIntegratorData()
    {
      dt = create_array<order>(0.0);
    }

    void
    update_dt(Number dt_new)
    {
      dt_backup = dt;

      for (int i = order - 2; i >= 0; i--)
        {
          dt[i + 1] = dt[i];
        }

      dt[0] = dt_new;

      update_weights();
    }

    void
    rollback()
    {
      dt = dt_backup;

      update_weights();
    }

    Number
    get_current_dt() const
    {
      return dt[0];
    }

    Number
    get_primary_weight() const
    {
      return weights[0];
    }

    const std::array<Number, order + 1> &
    get_weights() const
    {
      return weights;
    }

    unsigned int
    effective_order() const
    {
      return std::count_if(dt.begin(), dt.end(), [](const auto &v) { return v > 0; });
    }

  private:
    void
    update_weights()
    {
      if (order == 2 && dt[1] != 0)
        {
          weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
          weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
          weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
        }
      if (order == 1 || dt[0] == 0)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }
      else
        {
          AssertThrow(order < 3, ExcMessage("Not implemented"));
        }
    }

    std::array<Number, order>     dt;
    std::array<Number, order>     dt_backup;
    std::array<Number, order + 1> weights;
  };

  template <int dim, typename Number, typename VectorizedArrayType, int order>
  class BDFIntegrator
  {
  public:
    static constexpr int n_order = order;

    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    template <int n_comp>
    using CellIntegrator =
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>;

    template <int n_comp>
    using CellIntegratorValue =
      typename FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>::
        value_type;

    template <int n_comp>
    using TimeCellIntegrator =
      std::array<FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>,
                 order>;

    BDFIntegrator(const TimeIntegratorData<Number, order> &time_data)
      : time_data(time_data)
    {}

    template <int n_comp>
    void
    compute_time_derivative(VectorizedArrayType &             value_result,
                            CellIntegratorValue<n_comp>       val,
                            const TimeCellIntegrator<n_comp> &time_phi,
                            const unsigned int                index,
                            const unsigned int                q) const
    {
      const auto &weights = time_data.get_weights();

      value_result += val[index] * weights[0];
      for (unsigned int i = 0; i < time_data.effective_order(); i++)
        {
          const auto val_old = time_phi[i].get_value(q);

          value_result += val_old[index] * weights[i + 1];
        }
    }

    // Temporary function, works for order = 2 only
    void
    set_old_vectors_pointers(const BlockVectorType *old_solution,
                             const BlockVectorType *old_old_solution) const
    {
      old_solutions[0] = old_solution;
      old_solutions[1] = old_old_solution;
    }

    // Also made mutable temprorarily
    mutable std::array<const BlockVectorType *, order> old_solutions;

    template <int n_comp>
    TimeCellIntegrator<n_comp>
    create_cell_intergator(const CellIntegrator<n_comp> &cell_integrator) const
    {
      return create_array<order>(cell_integrator);
    }

  private:
    const TimeIntegratorData<Number, order> &time_data;
  };
} // namespace Sintering