
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/sintering/initial_values_cloud.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>

#include <iostream>

using namespace dealii;

using Number     = double;
using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim = 2;

  const unsigned int fe_degree = 1;

  const bool is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream                        pcout(std::cout, is_zero_rank);
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  FE_Q<dim>                                 fe(fe_degree);
  MappingQ<dim>                             mapping(1);
  QGauss<dim>                               quad(fe_degree + 1);
  DoFHandler<dim>                           dof_handler(tria);
  AffineConstraints<Number>                 constraint;

  const double width = 10.;

  Point<dim> bottom_left;
  Point<dim> top_right;
  for (unsigned int d = 0; d < dim; ++d)
    {
      top_right[d] = width;
    }

  // Mesh settings
  const unsigned int elements_per_interface  = 8;
  const double       interface_width         = 1.0;
  const bool         periodic                = true;
  const bool         with_initial_refinement = true;

  Sintering::create_mesh(tria,
                         bottom_left,
                         top_right,
                         interface_width,
                         elements_per_interface,
                         periodic,
                         with_initial_refinement);

  // setup DoFHandlers
  dof_handler.distribute_dofs(fe);

  // setup constraints
  constraint.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);

  // add periodic
  if (periodic)
    {
      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(
            dof_handler, 2 * d, 2 * d + 1, d, periodicity_vector);
        }

      DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                       constraint);
    }

  constraint.close();

  // Read particles
  std::stringstream iss;
  iss << "#x,y,z,r" << std::endl;
  iss << "0.0,0.0,0.0,1.5" << std::endl;
  iss << "10.0,0.0,0.0,1.5" << std::endl;
  iss << "0.0,10.0,0.0,1.5" << std::endl;
  iss << "10.0,10.0,0.0,1.5" << std::endl;
  iss << "5.0,5.0,0.0,1.5" << std::endl;

  const auto   particles                 = Sintering::read_particles<dim>(iss);
  const bool   minimize_order_parameters = true;
  const double interface_buffer_ratio    = 0.5;

  Sintering::InitialValuesCloud<dim> initial_solution(particles,
                                                      interface_width,
                                                      minimize_order_parameters,
                                                      interface_buffer_ratio);

  // set initial condition
  VectorType solution(initial_solution.n_components());

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler),
    dof_handler.get_communicator());

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      solution.block(c).reinit(partitioner);
    }
  solution.zero_out_ghost_values();

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      initial_solution.set_component(c);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               initial_solution,
                               solution.block(c));

      constraint.distribute(solution.block(c));
    }

  // Grain tracker settings
  const double       threshold_lower          = 1e-15;
  const double       threshold_upper          = 1.01;
  const double       buffer_distance_ratio    = 0.05;
  const bool         allow_new_grains         = false;
  const bool         greedy_init              = !minimize_order_parameters;
  const unsigned int op_offset                = 2;
  const unsigned int max_order_parameters_num = 5;

  GrainTracker::Tracker<dim, Number> grain_tracker(dof_handler,
                                                   tria,
                                                   greedy_init,
                                                   allow_new_grains,
                                                   max_order_parameters_num,
                                                   threshold_lower,
                                                   threshold_upper,
                                                   buffer_distance_ratio,
                                                   op_offset);

  solution.update_ghost_values();
  grain_tracker.initial_setup(solution);
  grain_tracker.print_current_grains(pcout, true);
  solution.zero_out_ghost_values();
}
