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


namespace dealii
{
  namespace DoFTools
  {
    template <int dim,
              int spacedim,
              typename SparsityPatternType,
              typename number>
    void
    make_sparsity_pattern(const DoFHandler<dim, spacedim> &dof_handler,
                          SparsityPatternType             &sparsity,
                          const AffineConstraints<number> &constraints,
                          const Quadrature<dim>           &quadrature,
                          const bool keep_constrained_dofs = true)
    {
      const auto &fe = dof_handler.get_fe();

      const auto compute_scalar_bool_dof_mask = [&quadrature,
                                                 &dof_handler](const auto &fe) {
        Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
        MappingQ1<dim, spacedim> mapping;
        FEValues<dim> fe_values(mapping, fe, quadrature, update_values);

        Triangulation<dim, spacedim> tria;
        GridGenerator::hyper_cube(tria);

        fe_values.reinit(tria.begin());
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
            {
              double sum = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                sum +=
                  fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
              if (sum != 0)
                bool_dof_mask(i, j) = true;
            }

        return bool_dof_mask;
      };

      Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

      const unsigned int n_components = fe.n_components();

      if (fe.n_components() == 1)
        {
          bool_dof_mask = compute_scalar_bool_dof_mask(fe);
        }
      else
        {
          const auto scalar_bool_dof_mask =
            compute_scalar_bool_dof_mask(fe.base_element(0));

          for (unsigned int i = 0; i < scalar_bool_dof_mask.size(0); ++i)
            for (unsigned int j = 0; j < scalar_bool_dof_mask.size(1); ++j)
              if (scalar_bool_dof_mask[i][j])
                for (unsigned ic = 0; ic < n_components; ++ic)
                  for (unsigned jc = 0; jc < n_components; ++jc)
                    bool_dof_mask[i * n_components + ic]
                                 [j * n_components + jc] = true;
        }

      std::vector<types::global_dof_index> dofs_on_this_cell(fe.dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(dofs_on_this_cell);

            constraints.add_entries_local_to_global(dofs_on_this_cell,
                                                    sparsity,
                                                    keep_constrained_dofs,
                                                    bool_dof_mask);
          }
    }

  } // namespace DoFTools
} // namespace dealii