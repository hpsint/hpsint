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

#include <deal.II/base/mpi.h>

#include <chrono>
#include <functional>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

double
run_measurement(const std::function<void()> &fu,
                const unsigned int           n_repetitions = 100)
{
  // warm up
  for (unsigned int i = 0; i < 10; ++i)
    fu();

#ifdef LIKWID_PERFMON
  const auto add_padding = [](const int value) -> std::string {
    if (value < 10)
      return "000" + std::to_string(value);
    if (value < 100)
      return "00" + std::to_string(value);
    if (value < 1000)
      return "0" + std::to_string(value);
    if (value < 10000)
      return "" + std::to_string(value);

    AssertThrow(false, dealii::StandardExceptions::ExcInternalError());

    return "";
  };

  static unsigned int likwid_counter = 0;

  const std::string likwid_label =
    "likwid_" + add_padding(likwid_counter); // TODO
  likwid_counter++;
#endif

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(likwid_label.c_str());
#endif

  const auto timer = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < n_repetitions; ++i)
    fu();

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(likwid_label.c_str());
#endif

  const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now() - timer)
                        .count() /
                      1e9;

  return time;
}