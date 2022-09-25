#pragma once

#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <pf-applications/base/fe_integrator.h>

#include <pf-applications/sintering/tools.h>

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

  protected:
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
  struct AdvectionCellDataDer
  {
    Table<3, Tensor<1, dim, VectorizedArrayType>> df_d_c_eta;
    Table<3, moment_t<dim, VectorizedArrayType>>  dt_d_c_eta;

    static constexpr unsigned int n_comp_der_c_eta = (dim == 3 ? 6 : 3);

    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    void
    reinit(const unsigned int n_segments,
           const unsigned int n_order_parameters,
           const unsigned int n_q_points)
    {
      df_d_c_eta.reinit({n_segments, n_order_parameters + 1, n_q_points});
      dt_d_c_eta.reinit({n_segments, n_order_parameters + 1, n_q_points});
    }

    void
    fill(const unsigned int                                  cell,
         const Table<2, std::shared_ptr<BlockVectorType>> &  der_c_eta,
         const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    {
      FECellIntegrator<dim, n_comp_der_c_eta, Number, VectorizedArrayType>
        phi_ft_der(matrix_free);

      AssertDimension(df_d_c_eta.size(0), der_c_eta.size(0));
      AssertDimension(df_d_c_eta.size(1), der_c_eta.size(1));

      phi_ft_der.reinit(cell);

      for (unsigned int s = 0; s < df_d_c_eta.size(0); ++s)
        for (unsigned int i = 0; i < df_d_c_eta.size(1); ++i)
          {
            phi_ft_der.read_dof_values_plain(*der_c_eta[s][i]);
            phi_ft_der.evaluate(EvaluationFlags::values |
                                EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi_ft_der.n_q_points; ++q)
              {
                const auto val = phi_ft_der.get_value(q);

                for (unsigned int d = 0; d < dim; ++d)
                  df_d_c_eta(s, i, q)[d] = val[d];

                if constexpr (moment_s<dim, VectorizedArrayType> == 1)
                  dt_d_c_eta(s, i, q) = val[dim];
                else
                  for (unsigned int d = 0;
                       d < moment_s<dim, VectorizedArrayType>;
                       ++d)
                    dt_d_c_eta(s, i, q)[d] = val[dim + d];
              }
          }
    }

    const Tensor<1, dim, VectorizedArrayType> &
    df_dc(const unsigned int i_segment, const unsigned int q) const
    {
      return df_d_c_eta(i_segment, 0, q);
    }

    const moment_t<dim, VectorizedArrayType> &
    dt_dc(const unsigned int i_segment, const unsigned int q) const
    {
      return dt_d_c_eta(i_segment, 0, q);
    }

    const Tensor<1, dim, VectorizedArrayType> &
    df_deta(const unsigned int i_segment,
            const unsigned int j_op,
            const unsigned int q) const
    {
      return df_d_c_eta(i_segment, j_op + 1, q);
    }

    const moment_t<dim, VectorizedArrayType> &
    dt_deta(const unsigned int i_segment,
            const unsigned int j_op,
            const unsigned int q) const
    {
      return dt_d_c_eta(i_segment, j_op + 1, q);
    }
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionMechanism
  {
  public:
    // Force, torque and grain volume
    static constexpr unsigned int n_comp_volume_force_torque =
      (dim == 3 ? 7 : 4);
    static constexpr unsigned int n_comp_der_c_eta = (dim == 3 ? 6 : 3);

    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

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
    resize_storage(const unsigned int n_segments,
                   const unsigned int n_order_parameters,
                   const bool         omit_zeroing_entries = false)
    {
      const unsigned int n_items = n_order_parameters + 1;

      const unsigned int old_n_segments = der_c_eta.size(0);
      const unsigned int old_n_items    = der_c_eta.size(1);

      // Switch to a better data structure, Table is not appropriate
      if (old_n_items != n_items || old_n_segments != n_segments)
        {
          der_c_eta.reinit({n_segments, n_items}, true);

          if (n_segments > old_n_segments)
            for (unsigned int i = old_n_segments; i < n_segments; ++i)
              for (unsigned int j = 0; j < old_n_items; ++j)
                {
                  if (der_c_eta[i][j] == nullptr)
                    der_c_eta[i][j] = std::make_shared<BlockVectorType>();

                  if (old_n_segments != 0)
                    der_c_eta[i][j]->reinit(*der_c_eta[0][0],
                                            omit_zeroing_entries);
                }

          if (n_items > old_n_items)
            for (unsigned int i = 0; i < n_segments; ++i)
              for (unsigned int j = old_n_items; j < n_items; ++j)
                {
                  if (der_c_eta[i][j] == nullptr)
                    der_c_eta[i][j] = std::make_shared<BlockVectorType>();

                  if (old_n_items != 0)
                    der_c_eta[i][j]->reinit(*der_c_eta[0][0],
                                            omit_zeroing_entries);
                }
        }
    }

    void
    reinit_derivatives(
      const unsigned int                                  cell,
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
    {
      current_cell_data_der.fill(cell, der_c_eta, matrix_free);
    }

    void
    reinit(
      const unsigned int                                  cell,
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free) const
    {
      current_cell_data.resize(n_active_order_parameters);

      current_cell_data_der.reinit(der_c_eta.size(0),
                                   n_active_order_parameters,
                                   matrix_free.get_n_q_points());

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
      // mr / op_cell_data.volume * cross_product(op_cell_data.torque, p);

      // Total advection velocity
      const auto v_adv = vt + vr;

      return v_adv;
    }

    Tensor<1, dim, VectorizedArrayType>
    get_velocity_derivative_eta(const unsigned int                     op_id_i,
                                const unsigned int                     op_id_j,
                                const unsigned int                     q,
                                const Point<dim, VectorizedArrayType> &r) const
    {
      const auto &op_cell_data = current_cell_data.at(op_id_i);

      const auto &df_etaj =
        current_cell_data_der.df_deta(op_id_i, op_id_j, q); // vector
      const auto &dt_etaj =
        current_cell_data_der.dt_deta(op_id_i, op_id_j, q); // vector

      // Translational velocity derivative
      const auto dvt_detaj = mt / op_cell_data.volume * df_etaj;

      // Get vector from the particle center to the current point
      const auto p = r - op_cell_data.rc;

      // Rotational velocity derivative
      const auto dvr_detaj =
        mr / op_cell_data.volume * cross_product(dt_etaj, p);

      // Total advection velocity derivative
      const auto dv_adv_detaj = dvt_detaj + dvr_detaj;

      return dv_adv_detaj;
    }

    Tensor<1, dim, VectorizedArrayType>
    get_velocity_derivative_c(const unsigned int                     op_id_i,
                              const unsigned int                     q,
                              const Point<dim, VectorizedArrayType> &r) const
    {
      const auto &op_cell_data = current_cell_data.at(op_id_i);

      // Translational velocity derivative
      const auto dvt_dc =
        mt / op_cell_data.volume * current_cell_data_der.df_dc(op_id_i, q);

      // Get vector from the particle center to the current point
      const auto p = r - op_cell_data.rc;

      // Rotational velocity derivative
      const auto dvr_dc =
        mr / op_cell_data.volume *
        cross_product(current_cell_data_der.dt_dc(op_id_i, q), p);

      // Total advection velocity derivative
      const auto dv_adv_dc = dvt_dc + dvr_dc;

      return dv_adv_dc;
    }

    void
    nullify_derivatives()
    {
      for (unsigned int i = 0; i < der_c_eta.size(0); ++i)
        for (unsigned int j = 0; j < der_c_eta.size(1); ++j)
          *der_c_eta[i][j] = 0;
    }

    void
    nullify_data(const unsigned int n_segments,
                 const unsigned int n_order_parameters)
    {
      n_active_order_parameters = n_order_parameters;

      grains_data.assign(n_comp_volume_force_torque * n_segments, 0);
    }

    Number *
    grain_data(const unsigned int grain_id, const unsigned int segment_id)
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data[n_comp_volume_force_torque * index];
    }

    const Number *
    grain_data(const unsigned int grain_id, const unsigned int segment_id) const
    {
      const unsigned int index =
        grain_tracker.get_grain_segment_index(grain_id, segment_id);

      return &grains_data[n_comp_volume_force_torque * index];
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

    bool
    enabled() const
    {
      return is_active;
    }

    const Table<2, std::shared_ptr<BlockVectorType>> &
    get_der_c_eta() const
    {
      return der_c_eta;
    }

    Table<2, std::shared_ptr<BlockVectorType>> &
    get_der_c_eta()
    {
      return der_c_eta;
    }

  private:
    mutable Tensor<1, dim, VectorizedArrayType> current_velocity_derivative;

    const bool   is_active;
    const double mt;
    const double mr;

    mutable std::vector<AdvectionCellData<dim, Number, VectorizedArrayType>>
      current_cell_data;

    mutable AdvectionCellDataDer<dim, Number, VectorizedArrayType>
      current_cell_data_der;

    const GrainTracker::Tracker<dim, Number> &grain_tracker;

    unsigned int n_active_order_parameters;

    std::vector<Number> grains_data;

    Table<2, std::shared_ptr<BlockVectorType>> der_c_eta;
  };
} // namespace Sintering