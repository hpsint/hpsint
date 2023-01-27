// Simple bar with tensile load via body force

#define DEBUG_NORM
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1
#define MAX_SINTERING_GRAINS 2

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/base/fe_integrator.h>

#include <pf-applications/lac/preconditioners.h>
#include <pf-applications/lac/solvers_linear.h>
#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/data_out.h>

#include <pf-applications/structural/operator_elastic_linear.h>
#include <pf-applications/structural/stvenantkirchhoff.h>
#include <pf-applications/structural/tools.h>

using namespace dealii;
using namespace Structural;

template <int dim,
          int fe_degree,
          int n_points_1D,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  void
  run(const bool         matrix_based,
      const bool         manual_constraints,
      const unsigned int n_refinements)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // geometry and mesh
    Point<dim> bottom_left;
    Point<dim> top_right;
    top_right[0] = 2.0;
    std::vector<unsigned int> subdivisions(dim, 2);
    for (unsigned int d = 1; d < dim; ++d)
      {
        top_right[d]    = 1.0;
        subdivisions[d] = 1;
      }

    // Material
    const double E  = 1.0;
    const double nu = 0.25;

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

    GridGenerator::subdivided_hyper_rectangle(
      tria, subdivisions, bottom_left, top_right, false);

    tria.refine_global(n_refinements);

    // left faces
    for (const auto &cell : tria.active_cell_iterators())
      {
        for (const auto &face : cell->face_iterators())
          if (std::abs(face->center()(0)) < 1e-8)
            face->set_boundary_id(1);
          else if (std::abs(face->center()(1)) < 1e-8)
            face->set_boundary_id(2);
          else if (dim == 3 && std::abs(face->center()(2)) < 1e-8)
            face->set_boundary_id(3);
      }

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ<dim> mapping(1);

    QGauss<dim> quad(n_points_1D);

    AffineConstraints<Number> constraints;

    std::function<void(const DoFHandler<dim> &dof_handler,
                       AffineConstraints<Number> &)>
      constraints_imposition = {};

    if (!manual_constraints)
      {
        // Apply DBC
        constraints_imposition =
          [&](const DoFHandler<dim> &    local_dof_handler,
              AffineConstraints<Number> &local_constraints) {
            DoFTools::make_zero_boundary_constraints(local_dof_handler,
                                                     1,
                                                     local_constraints);
          };
        constraints_imposition(dof_handler, constraints);
      }

    constraints.close();

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints, quad, additional_data);

    // External loading
    auto body_force_x =
      [](FECellIntegrator<dim, dim, Number, VectorizedArrayType> &phi,
         const unsigned int                                       q) {
        (void)phi;
        (void)q;

        Tensor<1, dim, VectorizedArrayType> value_result;
        value_result[0] = -0.5;

        return value_result;
      };

    using NonLinearOperator =
      LinearElasticOperator<dim, Number, VectorizedArrayType>;

    NonLinearOperator nonlinear_operator(E,
                                         nu,
                                         matrix_free,
                                         constraints,
                                         matrix_based,
                                         constraints_imposition,
                                         body_force_x);

    VectorType solution;

    // Initialize
    nonlinear_operator.initialize_dof_vector(solution);

    // Apply constraints manually
    if (manual_constraints)
      {
        AffineConstraints<Number> constraints_dbc;
        DoFTools::make_zero_boundary_constraints(dof_handler,
                                                 1,
                                                 constraints_dbc);
        constraints_dbc.close();

        for (unsigned int d = 0; d < dim; ++d)
          nonlinear_operator.attach_dirichlet_boundary_conditions(
            constraints_dbc, d);
      }

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOutWithRanges<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);

      for (unsigned int d = 0; d < dim; ++d)
        data_out.add_data_vector(dof_handler, solution.block(d), "u");

      solution.update_ghost_values();
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;

      std::cout << "Outputing at t = " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };

    // Initial configuration
    output_result(0);

    ReductionControl solver_control_l(1000, 1e-10, 1e-9);

    auto preconditioner = Preconditioners::create(nonlinear_operator, "AMG");

    LinearSolvers::GMRESData gmres_data;
    gmres_data.orthogonalization_strategy = "modified gram schmidt";

    auto linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
      NonLinearOperator,
      Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                    *preconditioner,
                                                    solver_control_l,
                                                    gmres_data);

    const unsigned int                          nl_max_iter = 100;
    const double                                nl_abs_tol  = 1e-9;
    const double                                nl_rel_tol  = 1e-6;
    NonLinearSolvers::NewtonSolverSolverControl statistics(nl_max_iter,
                                                           nl_abs_tol,
                                                           nl_rel_tol);

    const bool         newton_do_update             = true;
    const unsigned int newton_threshold_newton_iter = 100;
    const unsigned int newton_threshold_linear_iter = 100;
    const bool         newton_reuse_preconditioner  = true;
    const bool         newton_use_damping           = false;

    auto non_linear_solver =
      std::make_unique<NonLinearSolvers::DampedNewtonSolver<VectorType>>(
        statistics,
        NonLinearSolvers::NewtonSolverAdditionalData(
          newton_do_update,
          newton_threshold_newton_iter,
          newton_threshold_linear_iter,
          newton_reuse_preconditioner,
          newton_use_damping));

    non_linear_solver->reinit_vector = [&](auto &vector) {
      nonlinear_operator.initialize_dof_vector(vector);
    };

    non_linear_solver->residual = [&](const auto &src, auto &dst) {
      nonlinear_operator.evaluate_nonlinear_residual(dst, src);

      statistics.increment_residual_evaluations(1);
    };

    non_linear_solver->setup_jacobian = [&](const auto &current_u) {
      (void)current_u;
      nonlinear_operator.do_update();
    };

    non_linear_solver->setup_preconditioner = [&](const auto &current_u) {
      (void)current_u;
      preconditioner->do_update();
    };

    non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
      const unsigned int n_iterations = linear_solver->solve(dst, src);

      statistics.increment_linear_iterations(n_iterations);

      return n_iterations;
    };

    non_linear_solver->solve(solution);

    // Final configuration
    output_result(1);

    std::cout << "Solver type:           "
              << (matrix_based ? "matrix-based" : "matrix-free") << std::endl;
    std::cout << "Constraints:           "
              << (manual_constraints ? "manual" : "via AffineConstraints")
              << std::endl;
    std::cout << "Linear iterations:     " << statistics.n_linear_iterations()
              << std::endl;
    std::cout << "Non-linear iterations: " << statistics.n_newton_iterations()
              << std::endl;
    std::cout << "Residual evaluations:  "
              << statistics.n_residual_evaluations() << std::endl;
    std::cout << "n_refinements:         " << n_refinements << std::endl;
  }
};

int
main(int argc, char **argv)
{
  bool         matrix_based       = false;
  bool         manual_constraints = false;
  unsigned int n_refinements      = 2;

  for (int i = 1; i < argc; ++i)
    {
      const std::string flag = std::string(argv[i]);
      if (flag == "--matrix-based")
        matrix_based = true;
      else if (flag == "--manual-constraints")
        manual_constraints = true;
      else if (flag == "--refinements" && i < argc - 1)
        n_refinements = atoi(argv[++i]);
    }

  AssertThrow(n_refinements > 0,
              ExcMessage("The number of refinements has to be positive"));

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim         = 2;
  constexpr int fe_degree   = 1;
  constexpr int n_points_1D = fe_degree + 1;

  Test<dim, fe_degree, n_points_1D, double, VectorizedArray<double, 4>> runner;
  runner.run(matrix_based, manual_constraints, n_refinements);
}
