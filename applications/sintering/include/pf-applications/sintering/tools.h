
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

namespace Sintering
{
  enum class InitialRefine
  {
    None,
    Base,
    Full
  };

  std::vector<unsigned int>
  get_primes(unsigned int start, unsigned int end)
  {
    std::vector<unsigned int> primes;

    for (unsigned int i = start; i <= end; ++i)
      {
        // Skip 0 and 1 as they are
        if (i == 1 || i == 0)
          continue;

        bool is_prime = true;

        // Iterate to check if i is prime
        for (unsigned int j = 2; j <= i / 2; ++j)
          if (i % j == 0)
            {
              is_prime = false;
              break;
            }

        if (is_prime)
          primes.push_back(i);
      }
    return primes;
  }

  std::pair<unsigned int, unsigned int>
  decompose_to_prime_tuple(const unsigned int n_ref, const unsigned max_prime)
  {
    const auto primes = get_primes(2, max_prime);

    unsigned int optimal_prime      = 0;
    unsigned int n_refinements      = 0;
    unsigned int min_elements_delta = numbers::invalid_unsigned_int;
    for (const auto &p : primes)
      {
        const unsigned int s =
          static_cast<unsigned int>(std::ceil(std::log2(n_ref / p)));
        const unsigned int n_current = p * std::pow(2, s);
        const unsigned int current_delta =
          static_cast<unsigned int>(std::abs(static_cast<int>(n_current - n_ref)));

        if (current_delta < min_elements_delta)
          {
            min_elements_delta = current_delta;
            optimal_prime      = p;
            n_refinements      = s;
          }
      }

    return std::make_pair(optimal_prime, n_refinements);
  }

  template <typename Triangulation, int dim>
  unsigned int
  create_mesh(Triangulation &     tria,
              const Point<dim> &  bottom_left,
              const Point<dim> &  top_right,
              const double        interface_width,
              const unsigned int  elements_per_interface,
              const bool          periodic,
              const InitialRefine refine,
              const unsigned int  max_prime                         = 0,
              const double        max_level0_elements_per_interface = 1.0)
  {
    const auto domain_size = top_right - bottom_left;

    const double h_e = interface_width / elements_per_interface;

    const unsigned int n_refinements_interface =
      static_cast<unsigned int>(std::ceil(
        std::log2(elements_per_interface / max_level0_elements_per_interface)));

    std::vector<unsigned int> subdivisions(dim);
    for (unsigned int d = 0; d < dim; d++)
      {
        subdivisions[d] = static_cast<unsigned int>(
          std::ceil(domain_size[d] / h_e / std::pow(2, n_refinements_interface)));
      }

    // Further reduce the number of initial subdivisions
    unsigned int n_refinements_base = 0;
    if (max_prime > 0)
      {
        const unsigned int n_ref =
          *std::min_element(subdivisions.begin(), subdivisions.end());

        const auto         pair = decompose_to_prime_tuple(n_ref, max_prime);
        const unsigned int optimal_prime = pair.first;

        n_refinements_base = pair.second;

        for (unsigned int d = 0; d < dim; d++)
          {
            subdivisions[d] = static_cast<unsigned int>(
              std::ceil(static_cast<double>(subdivisions[d]) / n_ref * optimal_prime));
          }
      }

    unsigned int n_refinements = n_refinements_base + n_refinements_interface;

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

    if (refine == InitialRefine::Base)
      {
        if (n_refinements_base > 0)
          tria.refine_global(n_refinements_base);
        return n_refinements_interface;
      }
    else if (refine == InitialRefine::Full)
      {
        if (n_refinements > 0)
          tria.refine_global(n_refinements);
        return 0;
      }
    else
      {
        return n_refinements;
      }
  }
} // namespace Sintering