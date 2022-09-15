#pragma once

#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <pf-applications/grain_tracker/tracker.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  struct AdvectionCellDataBase
  {
    Point<dim, VectorizedArrayType>     rc;
    Tensor<1, dim, VectorizedArrayType> force;
    VectorizedArrayType                 volume{1};

    bool
    is_zero() const
    {
      return zero;
    }

  protected:
    bool zero{true};

    void
    fill(const unsigned int cell_id,
         const Point<dim> & rc_i,
         const Number *     fdata)
    {
      volume[cell_id] = fdata[0];

      for (unsigned int d = 0; d < dim; ++d)
        {
          rc[d][cell_id]    = rc_i[d];
          force[d][cell_id] = fdata[d + 1];
        }

      zero = false;
    }

    void
    nullify(const unsigned int cell_id)
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          rc[d][cell_id]    = 0;
          force[d][cell_id] = 0;
        }
      volume[cell_id] = 1; // To prevent division by zero

      zero = true;
    }
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  struct AdvectionCellData
  {};

  template <typename Number, typename VectorizedArrayType>
  struct AdvectionCellData<2, Number, VectorizedArrayType>
    : public AdvectionCellDataBase<2, Number, VectorizedArrayType>
  {
    static constexpr int dim = 2;

    VectorizedArrayType torque{0};

    void
    fill(const unsigned int cell_id,
         const Point<dim> & rc_i,
         const Number *     fdata)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::fill(cell_id,
                                                                    rc_i,
                                                                    fdata);

      torque[cell_id] = fdata[dim];
    }

    void
    nullify(const unsigned int cell_id)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::nullify(cell_id);

      torque[cell_id] = 0;
    }

    Tensor<1, dim, VectorizedArrayType>
    cross(const Tensor<1, dim, VectorizedArrayType> &r) const
    {
      Tensor<1, dim, VectorizedArrayType> p;
      p[0] = -r[1];
      p[1] = r[0];
      p *= torque;

      return p;
    }
  };

  template <typename Number, typename VectorizedArrayType>
  struct AdvectionCellData<3, Number, VectorizedArrayType>
    : public AdvectionCellDataBase<3, Number, VectorizedArrayType>
  {
    static constexpr int dim = 3;

    Tensor<1, dim, VectorizedArrayType> torque;

    void
    fill(const unsigned int cell_id,
         const Point<dim> & rc_i,
         const Number *     fdata)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::fill(cell_id,
                                                                    rc_i,
                                                                    fdata);

      for (unsigned int d = 0; d < dim; ++d)
        {
          torque[d][cell_id] = fdata[dim + d];
        }
    }

    void
    nullify(const unsigned int cell_id)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::nullify(cell_id);

      for (unsigned int d = 0; d < dim; ++d)
        {
          torque[d][cell_id] = 0;
        }
    }

    Tensor<1, dim, VectorizedArrayType>
    cross(const Tensor<1, dim, VectorizedArrayType> &r) const
    {
      return cross_product_3d(torque, r);
    }
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionMechanism
  {
  public:
    // Force, torque and grain volume
    static constexpr unsigned int n_comp_force_torque = (dim == 3 ? 6 : 3);
    static constexpr unsigned int n_comp_total        = (dim == 3 ? 7 : 4);

    AdvectionMechanism(const bool                                enable,
                       const double                              mt,
                       const double                              mr,
                       const GrainTracker::Tracker<dim, Number> &grain_tracker)
      : is_active(enable)
      , mt(mt)
      , mr(mr)
      , grain_tracker(grain_tracker)
    {}

    void
    reinit(
      const unsigned int                                  cell,
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
    {
      current_cell_data.resize(n_active_order_parameters);

      for (unsigned int op = 0; op < n_active_order_parameters; ++op)
        {
          unsigned int i = 0;

          for (; i < matrix_free.n_active_entries_per_cell_batch(cell); ++i)
            {
              const auto icell      = matrix_free.get_cell_iterator(cell, i);
              const auto cell_index = icell->global_active_cell_index();

              const unsigned int particle_id =
                grain_tracker.get_particle_index(op, cell_index);

              if (particle_id != numbers::invalid_unsigned_int)
                {
                  const auto grain_and_segment =
                    grain_tracker.get_grain_and_segment(op, particle_id);

                  const auto &rc_i =
                    grain_tracker.get_segment_center(grain_and_segment.first,
                                                     grain_and_segment.second);

                  current_cell_data[op].fill(
                    i,
                    rc_i,
                    grain_data(grain_and_segment.first,
                               grain_and_segment.second));
                }
              else
                {
                  current_cell_data[op].nullify(i);
                }
            }

          // Initialize the rest for padding
          for (; i < VectorizedArrayType::size(); ++i)
            current_cell_data[op].nullify(i);
        }
    }

    bool
    has_velocity(const unsigned int order_parameter_id) const
    {
      return !current_cell_data.at(order_parameter_id).is_zero();
    }

    Tensor<1, dim, VectorizedArrayType>
    get_velocity(const unsigned int                     order_parameter_id,
                 const Point<dim, VectorizedArrayType> &r) const
    {
      const auto &op_cell_data = current_cell_data.at(order_parameter_id);

      // Translational velocity
      const auto vt = mt / op_cell_data.volume * op_cell_data.force;

      // Get vector from the particle center to the current point
      const auto r_rc = r - op_cell_data.rc;

      // Rotational velocity
      const auto vr = mr / op_cell_data.volume * op_cell_data.cross(r_rc);

      // Total advection velocity
      const auto v_adv = vt + vr;

      return v_adv;
    }

    Tensor<1, dim, VectorizedArrayType>
    get_velocity_derivative(const unsigned int order_parameter_id,
                            const Point<dim, VectorizedArrayType> p) const
    {
      (void)order_parameter_id;
      (void)p;
      return current_velocity_derivative;
    }

    void
    nullify_data(const unsigned int n_segments,
                 const unsigned int n_order_parameters)
    {
      n_active_order_parameters = n_order_parameters;

      grains_data.assign(n_comp_total * n_segments, 0);
    }

    void
    nullify_data_derivatives(const unsigned int n_segments,
                             const unsigned int n_order_parameters)
    {
      AssertDimension(n_active_order_parameters, n_order_parameters);

      grains_data_derivatives.assign(n_comp_force_torque *
                                       (n_order_parameters + 1) * n_segments,
                                     0);
    }

    Number *
    grain_data(const unsigned int grain_id, const unsigned int segment_id)
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data[n_comp_total * index];
    }

    const Number *
    grain_data(const unsigned int grain_id, const unsigned int segment_id) const
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data[n_comp_total * index];
    }

    Number *
    grain_data_derivative(const unsigned int grain_id,
                          const unsigned int segment_id)
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data_derivatives[n_comp_force_torque *
                                      (n_active_order_parameters + 1) * index];
    }

    const Number *
    grain_data_derivative(const unsigned int grain_id,
                          const unsigned int segment_id) const
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data_derivatives[n_comp_force_torque *
                                      (n_active_order_parameters + 1) * index];
    }

    std::vector<Number> &
    get_grains_data()
    {
      return grains_data;
    }

    const std::vector<Number> &
    get_grains_data() const
    {
      return grains_data;
    }

    std::vector<Number> &
    get_grains_data_derivatives()
    {
      return grains_data_derivatives;
    }

    const std::vector<Number> &
    get_grains_data_derivatives() const
    {
      return grains_data_derivatives;
    }

    bool
    enabled() const
    {
      return is_active;
    }

  private:
    mutable Tensor<1, dim, VectorizedArrayType> current_velocity_derivative;

    const bool   is_active;
    const double mt;
    const double mr;

    mutable std::vector<AdvectionCellData<dim, Number, VectorizedArrayType>>
      current_cell_data;

    const GrainTracker::Tracker<dim, Number> &grain_tracker;

    unsigned int n_active_order_parameters;

    std::vector<Number> grains_data;
    std::vector<Number> grains_data_derivatives;
  };
} // namespace Sintering