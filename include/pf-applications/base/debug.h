#pragma once

#include <deal.II/base/mpi.h>

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

  std::ofstream &
  get_log()
  {
    const unsigned int rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    std::string rank_str = "";

    if (rank < 10)
      rank_str = "000" + std::to_string(rank);
    else if (rank < 100)
      rank_str = "00" + std::to_string(rank);
    else if (rank < 1000)
      rank_str = "0" + std::to_string(rank);
    else if (rank < 10000)
      rank_str = "" + std::to_string(rank);
    else
      AssertThrow(false, dealii::ExcNotImplemented());

    static std::ofstream myfile("temp_" + rank_str, std::ios::out);

    return myfile;
  }

  void
  log_with_barrier(const std::string &label)
  {
    auto &my_file = get_log();

    my_file << label << "::0" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    my_file << label << "::1" << std::endl;
  }

  void
  log_without_barrier(const std::string &label)
  {
    auto &my_file = get_log();

    my_file << label << std::endl;
  }

} // namespace debug