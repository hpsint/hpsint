#include <deal.II/numerics/rtree.h>

#include <iostream>

int
main()
{
  constexpr unsigned int dim = 2;

  using namespace dealii;

  namespace bgi = boost::geometry::index;

  using PP = std::pair<Point<dim>, Point<dim>>;

  // note: boost can not handle circles/balls
  std::vector<BoundingBox<dim>> boxes;
  boxes.emplace_back(PP{{-0.30, -0.30}, {-0.10, -0.10}});
  boxes.emplace_back(PP{{-0.30, +0.30}, {-0.10, +0.40}});
  boxes.emplace_back(PP{{+0.10, -0.30}, {+0.10, +0.30}});

  const auto tree = pack_rtree_of_indices(boxes);

  BoundingBox<dim> box(PP{{-0.25, -0.25}, {+0.25, +0.25}});

  for (const auto &i : tree | bgi::adaptors::queried(bgi::intersects(box)))
    std::cout << "Point p: " << i << std::endl;
}