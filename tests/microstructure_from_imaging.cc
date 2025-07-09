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

#include <pf-applications/tests/macro.h>
#include <pf-applications/tests/microstructure_tester.h>

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

  InitialValuesMicrostructureImaging initial_values(fstream,
                                                    interface_width,
                                                    interface_direction);

  // Settings
  const std::vector<unsigned int> subdivisions{5, 1};
  const bool                      periodic     = false;
  const bool                      print_grains = false;
  const std::string               base_name    = file_name.stem();

  Test::track_grains_for_microstructure(
    initial_values, base_name, subdivisions, periodic, print_grains);
}
