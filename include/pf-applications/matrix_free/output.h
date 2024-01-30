// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
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

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>

namespace dealii
{
  namespace MyMatrixFreeTools
  {
    template <int dim, typename Number, typename VectorizedArrayType>
    void
    add_mf_indices_vector(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      DataOut<dim> &                                      data_out,
      const std::string name_batch_index = "mf_cell_batch_index",
      const std::string name_lane_index  = "mf_lane_index")
    {
      Vector<float> mf_cell_batch_index(
        matrix_free.get_dof_handler().get_triangulation().n_active_cells());
      Vector<float> mf_lane_index(
        matrix_free.get_dof_handler().get_triangulation().n_active_cells());

      for (const auto &cell :
           matrix_free.get_dof_handler().active_cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          const auto cell_index = matrix_free.get_matrix_free_cell_index(cell);

          mf_cell_batch_index[cell->active_cell_index()] =
            cell_index / VectorizedArrayType::size();
          mf_lane_index[cell->active_cell_index()] =
            cell_index % VectorizedArrayType::size();
        }

      data_out.add_data_vector(mf_cell_batch_index, name_batch_index);
      data_out.add_data_vector(mf_lane_index, name_lane_index);
    }
  } // namespace MyMatrixFreeTools
} // namespace dealii