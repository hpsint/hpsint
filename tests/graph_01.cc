#include <deal.II/base/mpi.h>

#include <pf-applications/grain_tracker/distributed_stitching.h>

using namespace dealii;

int
main()
{
  const unsigned int N = 9;

  std::vector<std::tuple<unsigned int, unsigned int>> edges;

  edges.emplace_back(0, 1);
  edges.emplace_back(0, 4);
  edges.emplace_back(1, 7);
  edges.emplace_back(4, 7);
  edges.emplace_back(2, 5);
  edges.emplace_back(3, 8);

  const auto c = GrainTracker::connected_components(N, edges);

  for (const auto i : c)
    std::cout << i << " ";
  std::cout << std::endl;
}