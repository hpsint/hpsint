
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

template <int dim>
class AdvectSolution : public Function<dim>
{
public:
  AdvectSolution(const Point<dim> &                   direction,
                 const Point<dim> &                   bottom_left,
                 const Point<dim> &                   top_right,
                 std::shared_ptr<const Function<dim>> initial_values)
    : Function<dim>(1)
    , direction(direction)
    , bottom_left(bottom_left)
    , top_right(top_right)
    , initial_values(initial_values)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const final
  {
    Point<dim> offset(direction);
    offset *= this->get_time();

    Point<dim> q(p);
    q -= offset;

    // Periodicity
    for (unsigned int d = 0; d < dim; ++d)
      {
        while (q[d] < bottom_left[d])
          {
            q[d] += top_right[d];
          }
        while (q[d] > top_right[d])
          {
            q[d] -= top_right[d];
          }
      }

    return initial_values->value(q, component);
  }

private:
  const Point<dim>                     direction;
  const Point<dim>                     bottom_left;
  const Point<dim>                     top_right;
  std::shared_ptr<const Function<dim>> initial_values;
};

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

  const double length = 60;
  const double width  = 22.5;

  Point<dim>                bottom_left;
  Point<dim>                top_right;
  std::vector<unsigned int> subdivisions(dim, 10);
  top_right[0]    = length;
  subdivisions[0] = 26;
  for (unsigned int d = 1; d < dim; ++d)
    {
      top_right[d] = width;
    }

  // Mesh settings
  const double       interface_width = 2.;
  const bool         periodic        = true;
  const unsigned int n_refinements   = 3;

  Sintering::create_mesh(
    tria, bottom_left, top_right, subdivisions, periodic, n_refinements);

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
  iss << "0.0,11.25,11.25,7.5" << std::endl;
  iss << "15.0,11.25,11.25,7.5" << std::endl;
  iss << "30.0,11.25,11.25,7.5" << std::endl;
  iss << "45.0,11.25,11.25,7.5" << std::endl;
  iss << "60.0,11.25,11.25,7.5" << std::endl;

  const auto   particles                 = Sintering::read_particles<dim>(iss);
  const bool   minimize_order_parameters = true;
  const double interface_buffer_ratio    = 0.5;

  const auto initial_solution =
    std::make_shared<Sintering::InitialValuesCloud<dim>>(
      particles,
      interface_width,
      minimize_order_parameters,
      interface_buffer_ratio);

  Point<dim> direction;
  direction[0] = 1.0;
  AdvectSolution<dim> advect_solution(direction,
                                      bottom_left,
                                      top_right,
                                      initial_solution);

  // set initial condition
  VectorType solution(initial_solution->n_components());

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler),
    dof_handler.get_communicator());

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      solution.block(c).reinit(partitioner);
    }
  solution.zero_out_ghost_values();

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

  double t_start = 0;
  double t_end   = 15;
  double t_step  = 1;

  for (double t = t_start; t < t_end + 0.1 * t_step; t += t_step)
    {
      advect_solution.set_time(t);

      for (unsigned int c = 0; c < solution.n_blocks(); ++c)
        {
          initial_solution->set_component(c);

          VectorTools::interpolate(mapping,
                                   dof_handler,
                                   advect_solution,
                                   solution.block(c));

          constraint.distribute(solution.block(c));
        }

      try
        {
          solution.update_ghost_values();

          const auto [has_reassigned_grains, has_op_number_changed] =
            std::abs(t - t_start) < 1e-16 ?
              grain_tracker.initial_setup(
                solution, initial_solution->n_order_parameters()) :
              grain_tracker.track(solution, solution.n_blocks() - 2);

          pcout << "Time t = " << t << std::endl;
          grain_tracker.print_current_grains(pcout, true);
          if (std::abs(t - t_end) > 1e-16)
            {
              pcout << std::endl;
            }

          solution.zero_out_ghost_values();

          if (has_reassigned_grains)
            {
              pcout << "Grains have been reassigned" << std::endl;
            }

          if (has_op_number_changed)
            {
              pcout << "Number of order parameters has changed" << std::endl;
            }
        }
      catch (const GrainTracker::ExcGrainsInconsistency &ex)
        {
          pcout << "GrainTracker::ExcGrainsInconsistency detected!"
                << std::endl;
          grain_tracker.print_old_grains(pcout);

          AssertThrow(false, ExcMessage(ex.what()));
        }
    }
}
