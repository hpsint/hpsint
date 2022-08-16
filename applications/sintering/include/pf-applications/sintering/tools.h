
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace Sintering
{
  template <typename Triangulation, int dim>
  unsigned int
  create_mesh(Triangulation &    tria,
              const Point<dim> & bottom_left,
              const Point<dim> & top_right,
              const double       interface_width,
              const unsigned int elements_per_interface,
              const bool         periodic,
              const bool         with_initial_refinement,
              const double       max_level0_elements_per_interface = 1.0)
  {
    const auto   domain_size   = top_right - bottom_left;
    const double domain_width  = domain_size[0];
    const double domain_height = domain_size[1];

    const double h_e = interface_width / elements_per_interface;

    unsigned int       n_refinements = 0;
    const unsigned int base          = 2;
    for (double initial_elements_per_interface = elements_per_interface;
         initial_elements_per_interface > max_level0_elements_per_interface;
         initial_elements_per_interface /= base, ++n_refinements)
      ;

    const unsigned int initial_nx = static_cast<unsigned int>(
      domain_width / h_e / std::pow(2, n_refinements));
    const unsigned int initial_ny = static_cast<unsigned int>(
      domain_height / h_e / std::pow(2, n_refinements));

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = initial_nx;
    subdivisions[1] = initial_ny;
    if (dim == 3)
      {
        const double       domain_depth = domain_size[2];
        const unsigned int initial_nz   = static_cast<unsigned int>(
          domain_depth / h_e / std::pow(2, n_refinements));
        subdivisions[2] = initial_nz;
      }

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    pcout << "Create subdivided hyperrectangle [";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(top_right[i] - bottom_left[i]);

        if (i + 1 != dim)
          pcout << "x";
      }

    pcout << "] with " << std::to_string(n_refinements) << " refinements and ";
    for (unsigned int i = 0; i < dim; ++i)
      {
        pcout << std::to_string(subdivisions[i]);

        if (i + 1 != dim)
          pcout << "x";
      }

    pcout << " subdivisions" << std::endl << std::endl;


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

    if (with_initial_refinement && (n_refinements > 0))
      {
        tria.refine_global(n_refinements);
      }

    return n_refinements;
  }
} // namespace Sintering