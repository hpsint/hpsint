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

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/numerics/data_out.h>

using namespace dealii;

namespace Sintering
{
  namespace Postprocessors
  {
    template <typename T>
    struct BlockVectorWrapper;

    template <typename T>
    struct BlockVectorWrapper<std::vector<T>>
    {
      BlockVectorWrapper(const std::vector<T> &v)
        : data(v)
      {}

      const typename std::vector<T>::value_type &
      block(typename std::vector<T>::size_type i) const
      {
        return data[i];
      }

      typename std::vector<T>::size_type
      n_blocks() const
      {
        return data.size();
      }

      const std::vector<T> &data;

      using BlockType = T;
    };

    template <int dim, typename VectorType>
    DataOutWithRanges<dim>
    build_default_output(const DoFHandler<dim> &         dof_handler,
                         const VectorType &              solution,
                         const std::vector<std::string> &names,
                         const bool                      add_subdomains = true,
                         const bool higher_order_cells                  = false)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = higher_order_cells;

      DataOutWithRanges<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.set_flags(flags);

      for (unsigned int b = 0; b < solution.n_blocks(); ++b)
        if (b < names.size() && !names[b].empty())
          data_out.add_data_vector(solution.block(b), names[b]);

      // Output subdomain structure
      if (add_subdomains)
        {
          auto subdomain_id =
            dof_handler.get_triangulation().locally_owned_subdomain();
          if (subdomain_id == numbers::invalid_subdomain_id)
            subdomain_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

          Vector<float> subdomain(
            dof_handler.get_triangulation().n_active_cells());
          for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain[i] = subdomain_id;

          data_out.add_data_vector(subdomain, "subdomain");
        }

      return data_out;
    }
  } // namespace Postprocessors
} // namespace Sintering