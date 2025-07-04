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

#define SINTERING_DIM 2
#define MAX_SINTERING_GRAINS 2
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

#include <deal.II/base/conditional_ostream.h>

#include <pf-applications/sintering/free_energy.h>
#include <pf-applications/sintering/operator_sintering_generic.h>

#include <pf-applications/tests/arguments.h>
#include <pf-applications/tests/definitions.h>
#include <pf-applications/tests/macro.h>

#include <sstream>
#include <vector>

using namespace Sintering;

namespace Test
{
  template <template <int dim, typename Number, typename VectorizedArrayType>
            typename NonLinearOperator,
            template <typename VectorizedArrayType>
            typename FreeEnergy,
            typename Iterator>
  void
  run_with_operator(Iterator begin, Iterator end)
  {
    MyArguments my_args(begin, end);

    std::stringstream output;
    std::stringstream output_stats;

    runner<NonLinearOperator, FreeEnergy>(my_args.argc(),
                                          my_args.argv(),
                                          output,
                                          output_stats);

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    auto result = output_stats.str();

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        // Remove evetything before "Final statistics:"
        result.erase(0, result.find("Final statistics:"));

        // Remove trailing whitespace and newlines
        result.erase(result.find_last_not_of(" \n\r\t") + 1);

        // Remove linear iterations data
        const auto p0 = result.find("  - n linear iterations");
        const auto p1 = result.find_first_of("\n", p0);
        result.erase(p0, p1 - p0 + 1);

        const auto p2 = result.find("  - avg linear iterations");
        const auto p3 = result.find_first_of("\n", p2);
        result.erase(p2, p3 - p2 + 1);
      }

    pcout << result << std::endl;
  }

  template <typename Iterator>
  void
  run_sintering_operator_generic(Iterator begin, Iterator end)
  {
    run_with_operator<SinteringOperatorGeneric, FreeEnergy>(begin, end);
  }
} // namespace Test
