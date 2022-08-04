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

  template <typename Number>
  class TimeIntegratorData
  {
  public:
    TimeIntegratorData(unsigned int order)
      : dt(order)
      , dt_backup(order)
      , weights(order + 1)
    {}

    void
    update_dt(Number dt_new)
    {
      dt_backup = dt;

      for (int i = maximum_order() - 2; i >= 0; i--)
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

    const std::vector<Number> &
    get_weights() const
    {
      return weights;
    }

    unsigned int
    effective_order() const
    {
      return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
        return v > 0;
      });
    }

    unsigned int
    maximum_order() const
    {
      return dt.size();
    }


  private:
    void
    update_weights()
    {
      if (effective_order() == 3)
        {
          weights[1] = -(dt[0] + dt[1]) * (dt[0] + dt[1] + dt[2]) /
                       (dt[0] * dt[1] * (dt[1] + dt[2]));
          weights[2] =
            dt[0] * (dt[0] + dt[1] + dt[2]) / (dt[1] * dt[2] * (dt[0] + dt[1]));
          weights[3] = -dt[0] * (dt[0] + dt[1]) /
                       (dt[2] * (dt[1] + dt[2]) * (dt[0] + dt[1] + dt[2]));
          weights[0] = -(weights[1] + weights[2] + weights[3]);
        }
      else if (effective_order() == 2)
        {
          weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
          weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
          weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
        }
      else if (effective_order() == 1)
        {
          weights[0] = 1.0 / dt[0];
          weights[1] = -1.0 / dt[0];
        }
      else
        {
          AssertThrow(false, ExcMessage("Not implemented"));
        }
    }

    std::vector<Number> dt;
    std::vector<Number> dt_backup;
    std::vector<Number> weights;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class BDFIntegrator
  {
  public:
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
      std::vector<FECellIntegrator<dim, n_comp, Number, VectorizedArrayType>>;

    BDFIntegrator(const TimeIntegratorData<Number> &time_data)
      : time_data(time_data)
      , old_solutions(time_data.maximum_order())
    {
      for (unsigned int i = 0; i < old_solutions.size(); i++)
        old_solutions[i] = std::make_shared<BlockVectorType>();
    }

    template <int n_comp>
    void
    compute_time_derivative(VectorizedArrayType &             value_result,
                            CellIntegratorValue<n_comp>       val,
                            const TimeCellIntegrator<n_comp> &time_phi,
                            const unsigned int                index,
                            const unsigned int                q) const
    {
      AssertThrow(time_data.effective_order() == time_phi.size(),
                  ExcMessage("Inconsistent data structures provided!"));

      const auto &weights = time_data.get_weights();

      value_result += val[index] * weights[0];
      for (unsigned int i = 0; i < time_data.effective_order(); i++)
        {
          const auto val_old = time_phi[i].get_value(q);

          value_result += val_old[index] * weights[i + 1];
        }
    }

    template <int n_comp>
    TimeCellIntegrator<n_comp>
    create_cell_intergator(const CellIntegrator<n_comp> &cell_integrator) const
    {
      return TimeCellIntegrator<n_comp>(time_data.effective_order(),
                                        cell_integrator);
    }

    void
    commit_old_solutions() const
    {
      // Move pointers
      for (int i = old_solutions.size() - 2; i >= 0; i--)
        {
          old_solutions[i]->zero_out_ghost_values();
          *old_solutions[i + 1] = *old_solutions[i];
          old_solutions[i + 1]->update_ghost_values();
        }
    }

    void
    set_recent_old_solution(const BlockVectorType &src) const
    {
      *old_solutions[0] = src;
      old_solutions[0]->update_ghost_values();
    }

    const BlockVectorType &
    get_recent_old_solution() const
    {
      old_solutions[0]->zero_out_ghost_values();
      return *old_solutions[0];
    }

    void
    initialize_old_solutions(std::function<void(BlockVectorType &)> f)
    {
      for (unsigned int i = 1; i < old_solutions.size(); i++)
        f(*old_solutions[i]);
    }

    std::vector<std::shared_ptr<BlockVectorType>>
    get_old_solutions() const
    {
      std::vector<std::shared_ptr<BlockVectorType>> vec;
      for (unsigned int i = 1; i < old_solutions.size(); i++)
        {
          old_solutions[i]->zero_out_ghost_values();
          vec.push_back(old_solutions[i]);
        }

      return vec;
    }

    std::vector<std::shared_ptr<BlockVectorType>>
    get_old_solutions_all() const
    {
      return old_solutions;
    }

  private:
    const TimeIntegratorData<Number> &time_data;

    mutable std::vector<std::shared_ptr<BlockVectorType>> old_solutions;
  };
} // namespace Sintering