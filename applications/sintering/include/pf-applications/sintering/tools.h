
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace Sintering
{
  template <typename Triangulation, int dim>
  void
  create_mesh(Triangulation &    tria,
              const Point<dim> & bottom_left,
              const Point<dim> & top_right,
              const double       interface_width,
              const unsigned int elements_per_interface,
              const bool         periodic)
  {
    const auto   domain_size   = top_right - bottom_left;
    const double domain_width  = domain_size[0];
    const double domain_height = domain_size[1];

    const unsigned int initial_ny = 10;
    const unsigned int initial_nx =
      static_cast<unsigned int>(domain_width / domain_height * initial_ny);

    const unsigned int n_refinements = static_cast<unsigned int>(
      std::round(std::log2(elements_per_interface / interface_width *
                           domain_height / initial_ny)));

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = initial_nx;
    subdivisions[1] = initial_ny;
    if (dim == 3)
      {
        const double       domain_depth = domain_size[2];
        const unsigned int initial_nz =
          static_cast<unsigned int>(domain_depth / domain_height * initial_ny);
        subdivisions[2] = initial_nz;
      }

    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, true);

    if (periodic)
      {
        // Need to work with triangulation here
        std::vector<
          GridTools::PeriodicFacePair<typename Triangulation::cell_iterator>>
          periodicity_vector;

        for (unsigned int d = 0; d < dim; ++d)
          {
            GridTools::collect_periodic_faces(
              tria, 2 * d, 2 * d + 1, d, periodicity_vector);
          }

        tria.add_periodicity(periodicity_vector);
      }

    if (n_refinements > 0)
      {
        tria.refine_global(n_refinements);
      }
  }
} // namespace Sintering