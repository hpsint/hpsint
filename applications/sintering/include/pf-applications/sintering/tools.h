
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
    const auto domain_size = top_right - bottom_left;

    const double h_e = interface_width / elements_per_interface;
    const double min_size =
      *std::min_element(domain_size.begin_raw(), domain_size.end_raw());
    const unsigned int n_ref = static_cast<unsigned int>(min_size / h_e);

    const unsigned int              base = 2;
    const std::vector<unsigned int> primes{2, 3, 5, 7};

    unsigned int optimal_prime      = 0;
    unsigned int n_refinements      = 0;
    unsigned int min_elements_delta = numbers::invalid_unsigned_int;
    for (const auto &p : primes)
      {
        const unsigned int s =
          static_cast<unsigned int>(std::ceil(std::log2(n_ref / p)));
        const unsigned int n_current     = p * std::pow(base, s);
        const unsigned int current_delta = n_current - n_ref;

        if (current_delta < min_elements_delta)
          {
            min_elements_delta = current_delta;
            optimal_prime      = p;
            n_refinements      = s;
          }
      }

    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int d = 0; d < dim; d++)
      {
        subdivisions[d] = static_cast<unsigned int>(
          std::ceil(domain_size[d] / min_size * optimal_prime));
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