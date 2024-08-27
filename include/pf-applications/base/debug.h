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

#pragma once

#include <sstream>
#include <string>

#define AssertThrowDistributedDimension(size)                        \
  {                                                                  \
    const auto min_size = Utilities::MPI::min(size, MPI_COMM_WORLD); \
    const auto max_size = Utilities::MPI::max(size, MPI_COMM_WORLD); \
    AssertThrow(min_size == max_size,                                \
                ExcDimensionMismatch(min_size, max_size));           \
  }

namespace debug
{
  template <typename BlockVectorType, typename Stream>
  void
  print_vector(const BlockVectorType &vec,
               const std::string &    label,
               Stream &               stream)
  {
    const unsigned int n_len = vec.block(0).size();

    stream << label << ":" << '\n';
    for (unsigned int i = 0; i < n_len; ++i)
      {
        for (unsigned int b = 0; b < vec.n_blocks(); ++b)
          stream << vec.block(b)[i] << "  ";
        stream << '\n';
      }

    stream << '\n';
  }

  template <typename Vector>
  std::string
  to_string(const Vector &vec, std::string sep = ",")
  {
    std::stringstream ss;
    for (unsigned int i = 0; i < vec.size(); ++i)
      {
        if (i != 0)
          ss << sep;
        ss << vec[i];
      }
    return ss.str();
  }

  template <typename Iterator>
  std::string
  to_string(Iterator begin, Iterator end, std::string sep = ",")
  {
    std::stringstream ss;
    for (Iterator current = begin; current != end;)
      {
        ss << *current;
        if (++current != end)
          ss << sep;
      }
    return ss.str();
  }
} // namespace debug