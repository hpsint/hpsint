#include <deal.II/base/mpi.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm     comm    = MPI_COMM_WORLD;
  const unsigned int n_procs = Utilities::MPI::n_mpi_processes(comm);
  const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

  AssertDimension(n_procs, 4);
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

  const auto results = Utilities::MPI::gather(
    comm, GrainTracker::perform_distributed_stitching(comm, input), 0);

  if (my_rank == 0)
    for (const auto &result : results)
      {
        for (const auto i : result)
          std::cout << i << " ";
        std::cout << std::endl;
      }
}