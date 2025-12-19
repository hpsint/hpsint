// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <fstream>
#include <regex>
#include <string>

namespace hpsint
{
  using namespace dealii;

  template <typename Parameters>
  void
  parse_params(const int          argc,
               char             **argv,
               const unsigned int offset,
               Parameters        &params,
               std::ostream      &out)
  {
    ConditionalOStream pcout(out,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // Override params directly via command line
    for (unsigned int i = offset; i < static_cast<unsigned int>(argc); ++i)
      {
        const std::string flag = std::string(argv[i]);

        // The first entry can be a file with parameters
        if (i == offset && flag.substr(0, 2) != "--")
          {
            pcout << "Input parameters file:" << std::endl;
            pcout << std::ifstream(argv[offset]).rdbuf() << std::endl;

            params.parse(std::string(argv[offset]));
          }
        // Parse custom options
        else if (flag.substr(0, 2) == "--")
          {
            std::regex  rgx("--([(\\w*).]*)=\"?(.*)\"?");
            std::smatch matches;

            AssertThrow(std::regex_search(flag, matches, rgx),
                        ExcMessage("Incorrect parameter string specified: " +
                                   flag + "\nThe correct format is:\n" +
                                   "--Path.To.Option=\"value\"\nor\n" +
                                   "--Path.To.Option=value\n" +
                                   "if 'value' does not contain spaces"));

            const std::string param_name  = matches[1].str();
            const std::string param_value = matches[2].str();

            params.set(param_name, param_value);
          }
      }

    params.check();

    pcout << "Parameters in JSON format:" << std::endl;
    params.print_input(out);
    pcout << std::endl;
  }
} // namespace hpsint