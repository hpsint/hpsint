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

#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <pf-applications/sintering/operator_sintering_data.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>

namespace Sintering
{
  using namespace dealii;
  using namespace hpsint;

  template <int dim, typename Number, typename VectorizedArrayType>
  struct AdvectionCellDataBase
  {
    Point<dim, VectorizedArrayType>     rc;
    Tensor<1, dim, VectorizedArrayType> force;
    VectorizedArrayType                 volume_inv{-1.};

    bool
    has_non_zero() const
    {
      return std::any_of(volume_inv.begin(),
                         volume_inv.end(),
                         [](const auto &val) { return val > 0; });
    }

  protected:
    void
    fill(const unsigned int cell_id, const Number *rc_i, const Number *fdata)
    {
      volume_inv[cell_id] = 1.0 / fdata[0];

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
      volume_inv[cell_id] = -1.; // To prevent division by zero
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
    fill(const unsigned int cell_id, const Number *rc_i, const Number *fdata)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::fill(cell_id,
                                                                    rc_i,
                                                                    fdata);

      torque[cell_id] = fdata[dim + 1];
    }

    void
    nullify(const unsigned int cell_id)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::nullify(cell_id);

      torque[cell_id] = 0;
    }

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
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
    fill(const unsigned int cell_id, const Number *rc_i, const Number *fdata)
    {
      AdvectionCellDataBase<dim, Number, VectorizedArrayType>::fill(cell_id,
                                                                    rc_i,
                                                                    fdata);

      for (unsigned int d = 0; d < dim; ++d)
        {
          torque[d][cell_id] = fdata[dim + 1 + d];
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

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    cross(const Tensor<1, dim, VectorizedArrayType> &r) const
    {
      return cross_product_3d(torque, r);
    }
  };

  // Forward declaration
  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionMechanism;

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionVelocityData
  {
  public:
    AdvectionVelocityData()
    {}

    AdvectionVelocityData(
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data)
      : advection(advection)
      , sintering_data(sintering_data)
    {}

    AdvectionVelocityData(
      const unsigned int                                          cell,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data)
      : advection(advection)
      , sintering_data(sintering_data)
    {
      reinit(cell);
    }

    void
    reinit(const unsigned int cell)
    {
      grain_to_relevant_grain.clear();

      if (advection.enabled() && advection.get_index_ptr().size() > 0)
        {
          const unsigned int n_lanes = VectorizedArrayType::size();

          const auto n_indices =
            advection.cell_index_ptr(cell + 1) - advection.cell_index_ptr(cell);
          const auto n_op = n_indices / n_lanes;

          AssertDimension(n_indices % n_lanes, 0);

          current_cell_data.resize(n_op);

          for (unsigned int i = 0; i < n_indices; ++i)
            {
              const auto index =
                advection.cell_index_values(advection.cell_index_ptr(cell) + i);

              if (index != numbers::invalid_unsigned_int)
                current_cell_data[i / n_lanes].fill(
                  i % n_lanes,
                  advection.grain_center(index),
                  advection.grain_data(index));
              else
                current_cell_data[i / n_lanes].nullify(i % n_lanes);
            }


          has_velocity_vector.resize(current_cell_data.size());
          for (unsigned int op = 0; op < current_cell_data.size(); ++op)
            has_velocity_vector[op] = current_cell_data[op].has_non_zero();
        }

      if (sintering_data.cut_off_enabled())
        grain_to_relevant_grain =
          sintering_data.get_grain_to_relevant_grain(cell);
    }

    DEAL_II_ALWAYS_INLINE inline bool
    empty() const
    {
      return current_cell_data.empty() && has_velocity_vector.empty();
    }

    DEAL_II_ALWAYS_INLINE inline bool
    has_velocity(const unsigned int order_parameter_id) const
    {
      if (has_velocity_vector.empty())
        return false;

      const auto relevant_id = get_relevant_id(order_parameter_id);

      if (relevant_id == static_cast<unsigned char>(255))
        return false;

      AssertIndexRange(relevant_id, has_velocity_vector.size());

      return has_velocity_vector[relevant_id];
    }

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    get_velocity(const unsigned int                     order_parameter_id,
                 const Point<dim, VectorizedArrayType> &r) const
    {
      const auto relevant_id = get_relevant_id(order_parameter_id);

      // Translational velocity
      const auto vt = get_translation_velocity(relevant_id);

      // Rotational velocity
      const auto vr = get_rotation_velocity(relevant_id, r);

      // Total advection velocity
      const auto v_adv = vt + vr;

      return v_adv;
    }

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    get_translation_velocity(const unsigned int order_parameter_id) const
    {
      const auto relevant_id = get_relevant_id(order_parameter_id);

      const auto &op_cell_data = current_cell_data.at(relevant_id);

      return advection.calc_translation_velocity(op_cell_data);
    }

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    get_rotation_velocity(const unsigned int order_parameter_id,
                          const Point<dim, VectorizedArrayType> &r) const
    {
      const auto relevant_id = get_relevant_id(order_parameter_id);

      const auto &op_cell_data = current_cell_data.at(relevant_id);

      return advection.calc_rotation_velocity(op_cell_data, r);
    }

  private:
    DEAL_II_ALWAYS_INLINE inline unsigned char
    get_relevant_id(const unsigned int order_parameter_id) const
    {
      if (grain_to_relevant_grain.empty())
        return static_cast<unsigned char>(order_parameter_id);

      AssertIndexRange(order_parameter_id, grain_to_relevant_grain.size());

      return grain_to_relevant_grain[order_parameter_id];
    }

    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;

    const SinteringOperatorData<dim, VectorizedArrayType> &sintering_data;

    std::vector<AdvectionCellData<dim, Number, VectorizedArrayType>>
      current_cell_data;

    std::vector<bool> has_velocity_vector;

    std::vector<unsigned char> grain_to_relevant_grain;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionMechanism
  {
  public:
    // Force, torque and grain volume
    static constexpr unsigned int n_comp_volume_force_torque =
      (dim == 3 ? 7 : 4);

    AdvectionMechanism()
      : is_active(false)
      , mt(0.0)
      , mr(0.0)
    {}

    AdvectionMechanism(const bool enable, const double mt, const double mr)
      : is_active(enable)
      , mt(mt)
      , mr(mr)
    {}

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    calc_translation_velocity(
      const AdvectionCellData<dim, Number, VectorizedArrayType> &op_cell_data)
      const
    {
      // Translational velocity
      const auto vt = mt * op_cell_data.volume_inv * op_cell_data.force;

      return vt;
    }

    DEAL_II_ALWAYS_INLINE inline Tensor<1, dim, VectorizedArrayType>
    calc_rotation_velocity(
      const AdvectionCellData<dim, Number, VectorizedArrayType> &op_cell_data,
      const Point<dim, VectorizedArrayType>                     &r) const
    {
      // Get vector from the particle center to the current point
      const auto r_rc = r - op_cell_data.rc;

      // Rotational velocity
      const auto vr = mr * op_cell_data.volume_inv * op_cell_data.cross(r_rc);

      return vr;
    }

    Tensor<1, dim, Number>
    get_translation_velocity_for_grain(const unsigned int index) const
    {
      const Number *data = grain_data(index);

      const Number                 volume(*data++);
      const Tensor<1, dim, Number> force(make_array_view(data, data + dim));

      // Translational velocity
      const auto vt = mt * force / volume;

      return vt;
    }

    std::vector<unsigned int> &
    get_index_ptr()
    {
      return index_ptr;
    }

    std::vector<unsigned int> &
    get_index_values()
    {
      return index_values;
    }

    const std::vector<unsigned int> &
    get_index_ptr() const
    {
      return index_ptr;
    }

    const std::vector<unsigned int> &
    get_index_values() const
    {
      return index_values;
    }

    unsigned int
    cell_index_ptr(unsigned int index) const
    {
      return index_ptr[index];
    }

    unsigned int
    cell_index_values(unsigned int index) const
    {
      return index_values[index];
    }

    void
    nullify_data(const unsigned int n_segments)
    {
      grains_data.assign(n_comp_volume_force_torque * n_segments, 0);
      grains_center.assign(dim * n_segments, 0);

      index_ptr.clear();
      index_values.clear();

      index_ptr = {0};
    }

    Number *
    grain_data(const unsigned int index)
    {
      return &grains_data[n_comp_volume_force_torque * index];
    }

    const Number *
    grain_data(const unsigned int index) const
    {
      return &grains_data[n_comp_volume_force_torque * index];
    }

    Number *
    grain_center(const unsigned int index)
    {
      return &grains_center[dim * index];
    }

    const Number *
    grain_center(const unsigned int index) const
    {
      return &grains_center[dim * index];
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

    template <typename Stream>
    void
    print_forces(Stream                                   &out,
                 const GrainTracker::Tracker<dim, Number> &grain_tracker) const
    {
      out << std::endl;
      out << "Grains segments volumes, forces and torques:" << std::endl;

      for (const auto &[grain_id, grain] : grain_tracker.get_grains())
        {
          for (unsigned int segment_id = 0;
               segment_id < grain.get_segments().size();
               segment_id++)
            {
              const Number *data = grain_data(
                grain_tracker.get_grain_segment_index(grain_id, segment_id));

              Number                 volume(*data++);
              Tensor<1, dim, Number> force(make_array_view(data, data + dim));
              moment_t<dim, Number>  torque(
                create_moment_from_buffer<dim>(data + dim));

              out << "Grain id = " << grain_id
                  << ", segment id = " << segment_id << ": "
                  << "volume = " << volume << " | "
                  << "force  = " << force << " | "
                  << "torque = " << torque << std::endl;
            }
        }

      out << std::endl;
    }

  private:
    const bool   is_active;
    const double mt;
    const double mr;

    std::vector<unsigned int> index_ptr;
    std::vector<unsigned int> index_values;

    std::vector<Number> grains_data;
    std::vector<Number> grains_center;
  };
} // namespace Sintering