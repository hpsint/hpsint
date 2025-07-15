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

#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_q.h>

#include <pf-applications/sintering/advection.h>

#include <pf-applications/time_integration/time_schemes.h>

using namespace dealii;
using namespace Sintering;
using namespace TimeIntegration;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr unsigned int dim = 2;
  using Number               = double;
  using VectorizedArrayType  = VectorizedArray<Number, 2>;

  const unsigned int n_global_refinements = 2;
  const unsigned int fe_degree            = 1;
  FE_Q<dim>          fe(fe_degree);
  QGauss<dim>        quadrature(fe_degree + 1);
  MappingQ1<dim>     mapping;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_quadrature_points;
  additional_data.overlap_communication_computation = false;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, quadrature, additional_data);

  // Material properties
  const double kappa_c = 1;
  const double kappa_p = 0.5;
  const double Mvol    = 1e-2;
  const double Mvap    = 1e-10;
  const double Msurf   = 4;
  const double Mgb     = 0.4;
  const double L       = 1;
  const double mt      = 1.0;
  const double mr      = 1.0;

  const std::shared_ptr<MobilityProvider> mobility_provider =
    std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

  TimeIntegration::TimeIntegratorData<Number> time_data;

  SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
    kappa_c, kappa_p, mobility_provider, std::move(time_data));

  const unsigned int n_grains     = 3;
  const unsigned int n_components = n_grains + 2;
  sintering_data.set_n_components(n_components);

  // Dummy advection data
  AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism(true,
                                                                           mt,
                                                                           mr);

  // One segment per order param
  advection_mechanism.nullify_data(n_grains);

  // Set up indices
  const auto index_increment = n_grains * VectorizedArrayType::size();
  const auto n_index_values =
    n_grains * VectorizedArrayType::size() * matrix_free.n_cell_batches();

  for (unsigned int i = 0; i < matrix_free.n_cell_batches(); ++i)
    advection_mechanism.get_index_ptr().push_back(
      advection_mechanism.get_index_ptr().back() + index_increment);

  advection_mechanism.get_index_values().resize(n_index_values);
  for (unsigned int i = 0; i < n_grains; ++i)
    {
      auto gdata     = advection_mechanism.grain_data(i);
      gdata[0]       = 1.0 + (i + 1) * i; // volume
      gdata[1]       = 1.0 + i;           // force
      gdata[dim + 1] = 1.0 + i;           // torque

      // Center for i-th grain is at the point (i, i)
      auto cdata = advection_mechanism.grain_center(i);
      for (unsigned int d = 0; d < dim; ++d)
        cdata[d] = i;

      for (unsigned int j = 0; j < n_grains * matrix_free.n_physical_cells();
           j += index_increment)
        {
          const auto index_start = i * VectorizedArrayType::size() + j;
          std::fill(advection_mechanism.get_index_values().begin() +
                      index_start,
                    advection_mechanism.get_index_values().begin() +
                      index_start + VectorizedArrayType::size(),
                    i);
        }
    }

  std::cout << "index_ptr:    ";
  std::copy(advection_mechanism.get_index_ptr().begin(),
            advection_mechanism.get_index_ptr().end(),
            std::ostream_iterator<unsigned int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "index_values: ";
  std::copy(advection_mechanism.get_index_values().begin(),
            advection_mechanism.get_index_values().end(),
            std::ostream_iterator<unsigned int>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "grains_data:  ";
  std::copy(advection_mechanism.get_grains_data().begin(),
            advection_mechanism.get_grains_data().end(),
            std::ostream_iterator<Number>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "grain_center: ";
  std::copy(advection_mechanism.grain_center(0),
            advection_mechanism.grain_center(0) + n_grains * dim,
            std::ostream_iterator<Number>(std::cout, " "));
  std::cout << std::endl;

  std::cout << "dim              = " << dim << std::endl;
  std::cout << "n_grains         = " << n_grains << std::endl;
  std::cout << "n_cell_batches   = " << matrix_free.n_cell_batches()
            << std::endl;
  std::cout << "n_physical_cells = " << matrix_free.n_physical_cells()
            << std::endl;
  std::cout << "index_increment  = " << index_increment << std::endl;
  std::cout << "n_index_values   = " << n_index_values << std::endl;

  AdvectionVelocityData<dim, Number, VectorizedArrayType> advection_data(
    advection_mechanism, sintering_data);

  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      advection_data.reinit(cell);
      std::cout << "cell_batch = " << cell << ":" << std::endl;

      Point<dim, VectorizedArrayType> ref_point;

      for (unsigned int i = 0;
           i < matrix_free.n_active_entries_per_cell_batch(cell);
           ++i)
        {
          const auto icell  = matrix_free.get_cell_iterator(cell, i);
          const auto center = icell->barycenter();
          for (unsigned int d = 0; d < dim; ++d)
            ref_point[d][i] = center[d];
        }

      for (unsigned int op = 0; op < n_grains; ++op)
        {
          std::cout << "  op = " << op
                    << " | has_velocity = " << advection_data.has_velocity(op)
                    << " | trans_vel = "
                    << advection_data.get_translation_velocity(op)
                    << " | rot_vel = "
                    << advection_data.get_rotation_velocity(op, ref_point)
                    << " | tot_vel = "
                    << advection_data.get_velocity(op, ref_point) << std::endl;
        }
    }
}