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

#include <pf-applications/sintering/postprocessors.h>

void
Sintering::Postprocessors::write_table(const TableHandler &table,
                                       const double        t,
                                       const MPI_Comm     &comm,
                                       const std::string   save_path)
{
  if (Utilities::MPI::this_mpi_process(comm) != 0)
    return;

  const bool is_new = (t == 0);

  std::stringstream ss;
  table.write_text(ss);

  std::string line;

  std::ofstream ofs;
  ofs.open(save_path,
           is_new ? std::ofstream::out | std::ofstream::trunc :
                    std::ofstream::app);

  // Get header
  std::getline(ss, line);

  // Write header if we only start writing
  if (is_new)
    ofs << line << std::endl;

  // Take the data itself
  std::getline(ss, line);

  ofs << line << std::endl;
  ofs.close();
}
