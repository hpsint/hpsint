// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

// Test hpsint::is_domain_periodic() for domains periodic along none, some or
// all axes.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <pf-applications/grid/grid_tools.h>

#include <array>
#include <iostream>
#include <vector>

using namespace dealii;

template <int dim>
void
test(const std::array<bool, dim> &periodic_axes, ConditionalOStream &pcout)
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  Point<dim> p1, p2;
  for (unsigned int d = 0; d < dim; ++d)
    {
      p1[d] = 0.;
      p2[d] = 1. + d;
    }

  GridGenerator::hyper_rectangle(tria, p1, p2, true);

  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodicity_vector;
  for (unsigned int d = 0; d < dim; ++d)
    if (periodic_axes[d])
      GridTools::collect_periodic_faces(
        tria, 2 * d, 2 * d + 1, d, periodicity_vector);
  tria.add_periodicity(periodicity_vector);

  tria.refine_global(3);

  const auto result = hpsint::is_domain_periodic(tria);

  // All ranks must agree
  for (unsigned int d = 0; d < dim; ++d)
    AssertThrow(
      Utilities::MPI::min(static_cast<int>(result[d]), MPI_COMM_WORLD) ==
        Utilities::MPI::max(static_cast<int>(result[d]), MPI_COMM_WORLD),
      ExcMessage("Result differs between ranks"));

  pcout << "dim = " << dim << ", requested:";
  for (unsigned int d = 0; d < dim; ++d)
    pcout << ' ' << periodic_axes[d];
  pcout << ", detected:";
  for (unsigned int d = 0; d < dim; ++d)
    pcout << ' ' << result[d];
  pcout << std::endl;

  for (unsigned int d = 0; d < dim; ++d)
    AssertThrow(result[d] == periodic_axes[d],
                ExcMessage("Wrong periodicity detected"));
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const bool is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream pcout(std::cout, is_zero_rank);

  pcout << std::boolalpha;

  test<2>({{false, false}}, pcout);
  test<2>({{true, false}}, pcout);
  test<2>({{false, true}}, pcout);
  test<2>({{true, true}}, pcout);

  test<3>({{false, false, false}}, pcout);
  test<3>({{false, false, true}}, pcout);
  test<3>({{true, false, true}}, pcout);
  test<3>({{true, true, true}}, pcout);
}
