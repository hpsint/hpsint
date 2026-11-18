// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2026 by the hpsint authors
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
               const std::string     &label,
               Stream                &stream)
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

  namespace internal
  {
    // Number of components of tensors and points
    template <typename Vector>
    constexpr auto
    n_components(const Vector &, int) -> decltype(Vector::dimension)
    {
      return Vector::dimension;
    }

    // Number of components of containers
    template <typename Vector>
    auto
    n_components(const Vector &vec, long) -> decltype(vec.size())
    {
      return vec.size();
    }
  } // namespace internal

  /* Print components of a small vector (a tensor, a point or a container) in
   * a single line. Is not named print_vector since it would have the same
   * signature as the function above.
   */
  template <typename Vector, typename Stream>
  void
  print_components(const Vector &vec, const std::string &label, Stream &stream)
  {
    const unsigned int n_len = internal::n_components(vec, 0);

    stream << label << " = (";
    for (unsigned int i = 0; i < n_len; ++i)
      // Adding zero avoids printing negative zeros
      stream << (i > 0 ? ", " : "") << vec[i] + 0.;
    stream << ")" << '\n';
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