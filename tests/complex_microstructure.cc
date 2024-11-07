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

#include <pf-applications/sintering/initial_values_microstructure_imaging.h>

#include <pf-applications/tests/microstructure_tester.h>

#define STRING(x) #x
#define XSTRING(x) STRING(x)

using namespace Sintering;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  // Open microstructure file
  const std::string           microstructure = "81grains.micro";
  const std::filesystem::path source_path    = XSTRING(SOURCE_CODE_ROOT);
  const std::filesystem::path file_micro =
    source_path / "microstructures" / microstructure;

  std::ifstream fstream(file_micro);
  AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

  // Extract filename only without extension
  const std::filesystem::path file_name(microstructure);

  constexpr double             interface_width = 0.4;
  constexpr InterfaceDirection interface_direction =
    InterfaceDirection::outside;

  InitialValuesMicrostructure initial_values(fstream, interface_width);

  // Now build the mesh
  const unsigned int dim = 2;

  const unsigned int fe_degree      = 1;
  const unsigned int n_points_1D    = 2;
  const unsigned int n_subdivisions = 1;

  FE_Q<dim>      fe{fe_degree};
  MappingQ1<dim> mapping;
  Quadrature<1>  quad(QIterated<1>(QGauss<1>(n_points_1D), n_subdivisions));

  parallel::distributed::Triangulation<dim> tria(comm);

  const auto boundaries = initial_values.get_domain_boundaries();

  pcout << "n_grains    = " << initial_values.n_particles() << std::endl;
  pcout << "n_ops       = " << initial_values.n_order_parameters() << std::endl;
  pcout << "bottom_left = " << boundaries.first << std::endl;
  pcout << "top_right   = " << boundaries.second << std::endl;

  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {5, 1},
                                            boundaries.first,
                                            boundaries.second);

  const unsigned int n_refines_global = 4;
  const unsigned int n_refines_local  = 3;

  const double       top_fraction_of_cells    = 0.9;
  const double       bottom_fraction_of_cells = 0.1;
  const unsigned int max_refinement_depth     = 1;
  const unsigned int min_refinement_depth     = 3;
  const double       interface_val_min        = 0.05;
  const double       interface_val_max        = 0.95;
  const unsigned int op_offset                = 2;

  tria.refine_global(n_refines_global);

  const unsigned int n_global_levels_0 =
    tria.n_global_levels() + n_refines_local;

  // Settings
  const std::vector<unsigned int> subdivisions{5, 1};
  const bool                      periodic     = false;
  const bool                      print_grains = false;
  const std::string               base_name    = file_name.stem();

  Test::track_grains_for_microstructure(
    initial_values, base_name, subdivisions, periodic, print_grains);
}
