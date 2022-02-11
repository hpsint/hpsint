#pragma once

namespace dealii
{
  namespace VectorTools
  {
    template <typename VectorType>
    bool
    check_identity(VectorType &vec_0, VectorType &vec_1)
    {
      if (vec_0.get_partitioner()->locally_owned_size() !=
          vec_1.get_partitioner()->locally_owned_size())
        return false;

      for (unsigned int i = 0;
           i < vec_0.get_partitioner()->locally_owned_size();
           ++i)
        if (vec_0.local_element(i) != vec_1.local_element(i))
          return false;

      return true;
    }



    template <typename VectorType>
    void
    split_up_fast(const VectorType & vec,
                  VectorType &       vec_0,
                  VectorType &       vec_1,
                  const unsigned int n_grains)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          for (unsigned int j = 0; j < 2; ++j)
            {
              vec_0.local_element(i0++) = vec.local_element(i++);
            }

          for (unsigned int j = 0; j < n_grains; ++j)
            {
              vec_1.local_element(i1++) = vec.local_element(i++);
            }
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    split_up(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const VectorType &                                  vec,
             VectorType &                                        vec_0,
             VectorType &                                        vec_1)
    {
      vec.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            // read all dof_values
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());
            cell_all->get_dof_values(vec, local);

            // write Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(1));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c, j)];
                }

              cell.set_dof_values(local_component, vec_0);
            }

            // write Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_1);
            }
          }

      vec.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    split_up_fast(const VectorType & vec,
                  VectorType &       vec_0,
                  VectorType &       vec_1,
                  VectorType &       vec_2,
                  const unsigned int n_grains)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0, i2 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          vec_0.local_element(i0++) = vec.local_element(i++);
          vec_1.local_element(i1++) = vec.local_element(i++);

          for (unsigned int j = 0; j < n_grains; ++j)
            vec_2.local_element(i2++) = vec.local_element(i++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    split_up(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
             const VectorType &                                  vec,
             VectorType &                                        vec_0,
             VectorType &                                        vec_1,
             VectorType &                                        vec_2)
    {
      vec.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            // read all dof_values
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());
            cell_all->get_dof_values(vec, local);

            // write Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c, j)];
                }

              cell.set_dof_values(local_component, vec_0);
            }

            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 1,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_1);
            }

            // write Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local_component[i] =
                    local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                       j)];
                }

              cell.set_dof_values(local_component, vec_2);
            }
          }

      vec.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    merge_fast(const VectorType & vec_0,
               const VectorType & vec_1,
               VectorType &       vec,
               const unsigned int n_grains)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          for (unsigned int j = 0; j < 2; ++j)
            vec.local_element(i++) = vec_0.local_element(i0++);

          for (unsigned int j = 0; j < n_grains; ++j)
            vec.local_element(i++) = vec_1.local_element(i1++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    merge(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const VectorType &                                  vec_0,
          const VectorType &                                  vec_1,
          VectorType &                                        vec)
    {
      vec_0.update_ghost_values();
      vec_1.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());

            // read Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(1));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_0, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c, j)] =
                    local_component[i];
                }
            }

            // read Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_1, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read all dof_values
            cell_all->set_dof_values(local, vec);
          }

      vec_0.zero_out_ghost_values();
      vec_1.zero_out_ghost_values();
    }



    template <typename VectorType>
    void
    merge_fast(const VectorType & vec_0,
               const VectorType & vec_1,
               const VectorType & vec_2,
               VectorType &       vec,
               const unsigned int n_grains)
    {
      for (unsigned int i = 0, i0 = 0, i1 = 0, i2 = 0;
           i < vec.get_partitioner()->locally_owned_size();)
        {
          vec.local_element(i++) = vec_0.local_element(i0++);
          vec.local_element(i++) = vec_1.local_element(i1++);

          for (unsigned int j = 0; j < n_grains; ++j)
            vec.local_element(i++) = vec_2.local_element(i2++);
        }
    }



    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              typename VectorType>
    void
    merge(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const VectorType &                                  vec_0,
          const VectorType &                                  vec_1,
          const VectorType &                                  vec_2,
          VectorType &                                        vec)
    {
      vec_0.update_ghost_values();
      vec_1.update_ghost_values();
      vec_2.update_ghost_values();

      for (const auto &cell_all :
           matrix_free.get_dof_handler(0).active_cell_iterators())
        if (cell_all->is_locally_owned())
          {
            Vector<double> local(cell_all->get_fe().n_dofs_per_cell());

            // read Cahn-Hilliard components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_0, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c, j)] =
                    local_component[i];
                }
            }

            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(3 /*TODO*/));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_1, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 1,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read Allen-Cahn components
            {
              DoFCellAccessor<dim, dim, false> cell(
                &matrix_free.get_dof_handler(0).get_triangulation(),
                cell_all->level(),
                cell_all->index(),
                &matrix_free.get_dof_handler(2));

              Vector<double> local_component(cell.get_fe().n_dofs_per_cell());
              cell.get_dof_values(vec_2, local_component);

              for (unsigned int i = 0; i < cell.get_fe().n_dofs_per_cell(); ++i)
                {
                  const auto [c, j] =
                    cell.get_fe().system_to_component_index(i);

                  local[cell_all->get_fe().component_to_system_index(c + 2,
                                                                     j)] =
                    local_component[i];
                }
            }

            // read all dof_values
            cell_all->set_dof_values(local, vec);
          }

      vec_0.zero_out_ghost_values();
      vec_1.zero_out_ghost_values();
      vec_2.zero_out_ghost_values();
    }
  } // namespace VectorTools
} // namespace dealii