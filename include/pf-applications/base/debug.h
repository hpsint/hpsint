#pragma once

#include <string>

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
} // namespace debug