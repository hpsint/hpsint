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

#include <pf-applications/sintering/initial_values_microstructure_voronoi.h>

#include <pf-applications/tests/macro.h>
#include <pf-applications/tests/microstructure_tester.h>

using namespace dealii;
using namespace Sintering;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  // Open microstructure file
  const std::string           microstructure = "25grains.periodic";
  const std::filesystem::path source_path    = XSTRING(SOURCE_CODE_ROOT);
  const std::filesystem::path file_micro =
    source_path / "microstructures" / microstructure;

  std::ifstream fstream(file_micro);
  AssertThrow(fstream.is_open(), ExcMessage("File not found!"));

  // Extract filename only without extension
  const std::filesystem::path file_name(microstructure);

  constexpr double             interface_width     = 0.3;
  constexpr InterfaceDirection interface_direction = InterfaceDirection::inside;

  // Initial values settings
  InitialValuesMicrostructureVoronoi initial_values(fstream,
                                                    interface_width,
                                                    interface_direction);

  // Settings
  const std::vector<unsigned int> subdivisions{2, 2};
  const bool                      periodic     = true;
  const bool                      print_grains = true;
  const std::string               base_name    = file_name.stem();
  const auto boundaries = std::make_pair(Point<2>(0, 0), Point<2>(32, 32));

  Test::track_grains_for_microstructure(initial_values,
                                        base_name,
                                        boundaries,
                                        subdivisions,
                                        periodic,
                                        print_grains);
}
