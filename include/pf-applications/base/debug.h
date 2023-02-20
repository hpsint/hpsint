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
  to_string(const Vector &vec)
  {
    std::stringstream ss;
    for (unsigned int i = 0; i < vec.size(); ++i)
      {
        if (i != 0)
          ss << ",";
        ss << vec[i];
      }
    return ss.str();
  }
} // namespace debug