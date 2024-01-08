#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include <pf-applications/sintering/initial_values_cloud.h>

#include <filesystem>

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  AssertThrow(argc > 1, ExcMessage("At least one filename has to be provided"));

  const unsigned int dim = 3;

  for (int i = 1; i < argc; ++i)
    {
      const auto file_name = std::string(argv[i]);
      const auto stem_name =
        std::string(std::filesystem::path(file_name).stem());
      const auto parent_path = std::filesystem::path(file_name).parent_path();

      std::ifstream fstream(file_name);

      const auto particles = Sintering::read_particles<dim>(fstream);

      const double interface_width           = 0.0;
      const bool   minimize_order_parameters = true;
      const double interface_buffer_ratio    = 0.0;

      const auto initial_solution =
        std::make_shared<Sintering::InitialValuesCloud<dim>>(
          particles,
          interface_width,
          minimize_order_parameters,
          interface_buffer_ratio);

      const auto domain_boundaries = initial_solution->get_domain_boundaries();
      const auto order_parameter_to_grains =
        initial_solution->get_order_parameter_to_grains();

      std::vector<unsigned int> particle_to_order_parameter(particles.size());

      for (const auto i : order_parameter_to_grains)
        for (const auto j : i.second)
          particle_to_order_parameter[j] = i.first;

      Triangulation<dim> tria;
      GridGenerator::hyper_rectangle(tria,
                                     domain_boundaries.first,
                                     domain_boundaries.second);

      MappingQ1<dim> mapping;

      Particles::ParticleHandler particle_handler(tria, mapping, 3);

      std::vector<Point<dim>>          points;
      std::vector<std::vector<double>> properties;

      for (unsigned int p = 0; p < particles.size(); ++p)
        {
          points.push_back(particles[p].center);
          properties.push_back(std::vector<double>(
            {static_cast<double>(p),
             particles[p].radius,
             static_cast<double>(particle_to_order_parameter[p])}));
        }

      std::vector<std::vector<BoundingBox<dim>>> global_bounding_boxes(1);
      global_bounding_boxes[0].emplace_back(domain_boundaries);

      particle_handler.insert_global_particles(points,
                                               global_bounding_boxes,
                                               properties);

      {
        DataOut<dim> data_out;
        data_out.attach_triangulation(tria);
        data_out.build_patches();

        const auto fname = "my_grid_" + std::to_string(i - 1) + ".vtu";
        data_out.write_vtu_in_parallel(fname, MPI_COMM_WORLD);
      }

      {
        Particles::DataOut<dim> data_out;
        data_out.build_patches(
          particle_handler,
          std::vector<std::string>{"grain_id", "radius", "order_parameter"},
          std::vector<DataComponentInterpretation::DataComponentInterpretation>(
            3, DataComponentInterpretation::component_is_scalar));

        const auto fname = "my_particles_" + std::to_string(i - 1) + ".vtu";
        data_out.write_vtu_in_parallel(fname, MPI_COMM_WORLD);
      }
    }
}
