#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

using namespace dealii;

std::vector<unsigned int>
perform_distributed_stitching(
  const MPI_Comm                                                   comm,
  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input)
{
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  const unsigned int local_size = input.size();
  unsigned int       offset     = 0;

  MPI_Exscan(&local_size, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

  using T = std::vector<
    std::tuple<unsigned int,
               std::vector<std::tuple<unsigned int, unsigned int>>>>;

  while (true)
    {
      std::map<unsigned int, T> data_to_send;

      for (unsigned int i = 0; i < input.size(); ++i)
        {
          const auto input_i = input[i];
          for (unsigned int j = 0; j < input_i.size(); ++j)
            {
              const unsigned int other_rank = std::get<0>(input_i[j]);

              if (other_rank == my_rank)
                continue;

              std::vector<std::tuple<unsigned int, unsigned int>> temp;

              temp.emplace_back(my_rank, i + offset);

              for (unsigned int k = 0; k < input_i.size(); ++k)
                if (k != j)
                  temp.push_back(input_i[k]);

              std::sort(temp.begin(), temp.end());

              data_to_send[other_rank].emplace_back(std::get<1>(input_i[j]),
                                                    temp);
            }
        }

      bool finished = true;

      Utilities::MPI::ConsensusAlgorithms::selector<T>(
        [&]() {
          std::vector<unsigned int> targets;
          for (const auto i : data_to_send)
            targets.emplace_back(i.first);
          return targets;
        }(),
        [&](const unsigned int other_rank) { return data_to_send[other_rank]; },
        [&](const unsigned int, const auto &data) {
          for (const auto &data_i : data)
            {
              const unsigned int index   = std::get<0>(data_i) - offset;
              const auto &       values  = std::get<1>(data_i);
              auto &             input_i = input[index];

              const unsigned int old_size = input_i.size();

              input_i.insert(input_i.end(), values.begin(), values.end());
              std::sort(input_i.begin(), input_i.end());
              input_i.erase(std::unique(input_i.begin(), input_i.end()),
                            input_i.end());

              const unsigned int new_size = input_i.size();

              finished &= (old_size == new_size);
            }
        },
        comm);

      if (finished)
        break;
    }

  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input_valid;

  for (unsigned int i = 0; i < input.size(); ++i)
    {
      auto input_i = input[i];

      if (input_i.size() == 0)
        {
          std::vector<std::tuple<unsigned int, unsigned int>> temp;
          temp.emplace_back(my_rank, i + offset);
          input_valid.push_back(temp);
        }
      else
        {
          if ((my_rank <= std::get<0>(input_i[0])) &&
              ((i + offset) < std::get<1>(input_i[0])))
            {
              input_i.insert(
                input_i.begin(),
                std::tuple<unsigned int, unsigned int>{my_rank, i + offset});
              input_valid.push_back(input_i);
            }
        }
    }

  const unsigned int local_size_p = input_valid.size();
  unsigned int       offset_p     = 0;

  MPI_Exscan(&local_size_p, &offset_p, 1, MPI_UNSIGNED, MPI_SUM, comm);

  using U = std::vector<std::tuple<unsigned int, unsigned int>>;
  std::map<unsigned int, U> data_to_send_;

  for (unsigned int i = 0; i < input_valid.size(); ++i)
    {
      for (const auto j : input_valid[i])
        data_to_send_[std::get<0>(j)].emplace_back(std::get<1>(j),
                                                   i + offset_p);
    }

  std::vector<unsigned int> result(input.size(), numbers::invalid_unsigned_int);

  Utilities::MPI::ConsensusAlgorithms::selector<U>(
    [&]() {
      std::vector<unsigned int> targets;
      for (const auto i : data_to_send_)
        targets.emplace_back(i.first);
      return targets;
    }(),
    [&](const unsigned int other_rank) { return data_to_send_[other_rank]; },
    [&](const unsigned int, const auto &data) {
      for (const auto &i : data)
        {
          AssertDimension(result[std::get<0>(i) - offset],
                          numbers::invalid_unsigned_int);
          result[std::get<0>(i) - offset] = std::get<1>(i);
        }
    },
    comm);

  return result;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  AssertDimension(n_procs, 5);
  (void)n_procs;

  std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input;

  if (my_rank == 0)
    {
      input.resize(1);
      input[0].emplace_back(1, 1);
      input[0].emplace_back(2, 3);
    }
  else if (my_rank == 1)
    {
      input.resize(2);
      input[0].emplace_back(0, 0);
      input[0].emplace_back(3, 4);
      input[1].emplace_back(3, 5);
    }
  else if (my_rank == 2)
    {
      input.resize(1);
      input[0].emplace_back(0, 0);
      input[0].emplace_back(3, 4);
    }
  else if (my_rank == 3)
    {
      input.resize(3);
      input[0].emplace_back(1, 1);
      input[0].emplace_back(2, 3);
      input[1].emplace_back(1, 2);
    }

  const auto results =
    Utilities::MPI::gather(comm, perform_distributed_stitching(comm, input), 0);

  if (my_rank == 0)
    for (const auto &result : results)
      {
        for (const auto i : result)
          std::cout << i << " ";
        std::cout << std::endl;
      }
}