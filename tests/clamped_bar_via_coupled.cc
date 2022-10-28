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
#include <pf-applications/lac/solvers_nonlinear.h>

#include <pf-applications/numerics/data_out.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/initial_values.h>
#include <pf-applications/sintering/operator_sintering_coupled.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/structural/stvenantkirchhoff.h>
#include <pf-applications/structural/tools.h>

using namespace dealii;
using namespace Structural;
using namespace Sintering;

template <int dim>
class InitialValuesTest : public InitialValues<dim>
{
public:
  double
  do_value(const dealii::Point<dim> &p,
           const unsigned int        component) const final
  {
    std::vector<double> vals;

    double tol = 1e-9;

    if constexpr (dim == 2)
      {
        dealii::Point<dim> p1(0, 0);
        dealii::Point<dim> p2(1, 0);
        dealii::Point<dim> p3(2, 0);
        dealii::Point<dim> p4(0, 1);
        dealii::Point<dim> p5(1, 1);
        dealii::Point<dim> p6(2, 1);

        if (p.distance(p1) < tol)
          vals = {1.0, 0.1, 1.0, 0.0};
        else if (p.distance(p2) < tol)
          vals = {0.8, 0.2, 0.4, 0.4};
        else if (p.distance(p3) < tol)
          vals = {1.0, 0.3, 0.0, 1.0};
        else if (p.distance(p4) < tol)
          vals = {1.0, 0.4, 1.0, 0.0};
        else if (p.distance(p5) < tol)
          vals = {0.8, 0.5, 0.4, 0.4};
        else if (p.distance(p6) < tol)
          vals = {1.0, 0.6, 0.0, 1.0};
        else
          {
            std::cout << "Point = " << p << std::endl;
            throw std::runtime_error("Wrong point!");
          }
      }
    else
      {
        dealii::Point<dim> p1(0, 0, 0);
        dealii::Point<dim> p2(1, 0, 0);
        dealii::Point<dim> p3(2, 0, 0);
        dealii::Point<dim> p4(0, 1, 0);
        dealii::Point<dim> p5(1, 1, 0);
        dealii::Point<dim> p6(2, 1, 0);
        dealii::Point<dim> p7(0, 0, 1);
        dealii::Point<dim> p8(1, 0, 1);
        dealii::Point<dim> p9(2, 0, 1);
        dealii::Point<dim> p10(0, 1, 1);
        dealii::Point<dim> p11(1, 1, 1);
        dealii::Point<dim> p12(2, 1, 1);

        if (p.distance(p1) < tol || p.distance(p7) < tol)
          vals = {1.0, 0.1, 1.0, 0.0};
        else if (p.distance(p2) < tol || p.distance(p8) < tol)
          vals = {0.8, 0.2, 0.4, 0.4};
        else if (p.distance(p3) < tol || p.distance(p9) < tol)
          vals = {1.0, 0.3, 0.0, 1.0};
        else if (p.distance(p4) < tol || p.distance(p10) < tol)
          vals = {1.0, 0.4, 1.0, 0.0};
        else if (p.distance(p5) < tol || p.distance(p11) < tol)
          vals = {0.8, 0.5, 0.4, 0.4};
        else if (p.distance(p6) < tol || p.distance(p12) < tol)
          vals = {1.0, 0.6, 0.0, 1.0};
        else
          {
            std::cout << "Point = " << p << std::endl;
            throw std::runtime_error("Wrong point!");
          }
      }

    return component < 4 ? vals[component] : 0.0;
  }

  std::pair<dealii::Point<dim>, dealii::Point<dim>>
  get_domain_boundaries() const final
  {
    AssertThrow(false, ExcNotImplemented());
  }

  double
  get_r_max() const final
  {
    AssertThrow(false, ExcNotImplemented());
  }

  double
  get_interface_width() const final
  {
    AssertThrow(false, ExcNotImplemented());
  }

  unsigned int
  n_order_parameters() const final
  {
    return 2;
  }
};

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
  run(const bool matrix_based)
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

    const unsigned int n_refinements = 0;

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
    constraints.close();

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags =
      update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints, quad, additional_data);

    // Additional objects
    const double       A                      = 1.;
    const double       B                      = 1.;
    const double       kappa_c                = 0.1;
    const double       kappa_p                = 0.1;
    const unsigned int time_integration_order = 1;
    const unsigned int n_initial_components   = 4;

    std::shared_ptr<MobilityProvider> mobility_provider =
      std::make_shared<ProviderAbstract>(1e-3, 1e-10, 1e-2, 1e-3, 1.);

    SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
      A, B, kappa_c, kappa_p, mobility_provider, time_integration_order);

    sintering_data.set_n_components(n_initial_components);

    // Grain tracker settings
    const double       threshold_lower          = 1e-15;
    const double       threshold_upper          = 1.01;
    const double       buffer_distance_ratio    = 0.05;
    const bool         allow_new_grains         = false;
    const bool         greedy_init              = false;
    const unsigned int op_offset                = 2;
    const unsigned int max_order_parameters_num = 2;

    GrainTracker::Tracker<dim, Number> grain_tracker(dof_handler,
                                                     tria,
                                                     greedy_init,
                                                     allow_new_grains,
                                                     max_order_parameters_num,
                                                     threshold_lower,
                                                     threshold_upper,
                                                     buffer_distance_ratio,
                                                     op_offset);

    const bool   enable = false;
    const double mt     = 0.;
    const double mr     = 0.;

    AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism(
      enable, mt, mr, grain_tracker);

    // set initial condition
    InitialValuesTest<dim> initial_solution;

    TimeIntegration::SolutionHistory<VectorType> solution_history(2);

    // External loading
    auto body_force_x = [](const Point<dim, VectorizedArrayType> &p) {
      (void)p;

      Tensor<1, dim, VectorizedArrayType> value_result;
      value_result[0] = -0.5;

      return value_result;
    };

    using NonLinearOperator =
      SinteringOperatorCoupled<dim, Number, VectorizedArrayType>;

    // Elastic material properties
    double E  = 1;
    double nu = 0.25;

    NonLinearOperator nonlinear_operator(matrix_free,
                                         constraints,
                                         sintering_data,
                                         solution_history,
                                         advection_mechanism,
                                         E,
                                         nu,
                                         matrix_based,
                                         body_force_x);

    std::function<void(VectorType &)> f_init =
      [&nonlinear_operator](VectorType &v) {
        nonlinear_operator.initialize_dof_vector(v);
      };

    solution_history.apply(f_init);

    VectorType &solution = solution_history.get_current_solution();

    const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_relevant_dofs(dof_handler),
      dof_handler.get_communicator());

    for (unsigned int c = 0; c < solution.n_blocks(); ++c)
      solution.block(c).reinit(partitioner);

    solution.zero_out_ghost_values();

    for (unsigned int c = 0; c < sintering_data.n_components(); ++c)
      {
        initial_solution.set_component(c);

        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 initial_solution,
                                 solution.block(c));

        constraints.distribute(solution.block(c));
      }

    sintering_data.fill_quadrature_point_values(matrix_free, solution, enable);

    const auto output_result = [&](const double t) {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOutWithRanges<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(dof_handler, solution.block(0), "c");
      data_out.add_data_vector(dof_handler, solution.block(1), "mu");

      for (unsigned int ig = 2; ig < sintering_data.n_components(); ++ig)
        data_out.add_data_vector(dof_handler,
                                 solution.block(ig),
                                 "eta" + std::to_string(ig - 2));

      for (unsigned int d = 0; d < dim; ++d)
        data_out.add_data_vector(
          dof_handler, solution.block(sintering_data.n_components() + d), "u");

      solution.update_ghost_values();
      data_out.build_patches(mapping, fe_degree);

      static unsigned int counter = 0;

      std::cout << "Outputing at t = " << t << std::endl;

      std::string output = "solution." + std::to_string(counter++) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

      solution.zero_out_ghost_values();
    };

    // Initial configuration
    output_result(0);

    solution_history.set_recent_old_solution(solution);
    std::vector<Number> dts(time_integration_order);
    dts[0] = 1e-5;
    sintering_data.time_data.set_all_dt(dts);

    ReductionControl solver_control_l(100, 1e-10, 1e-9);

    auto preconditioner = Preconditioners::create(nonlinear_operator, "ILU");

    auto linear_solver = std::make_unique<LinearSolvers::SolverGMRESWrapper<
      NonLinearOperator,
      Preconditioners::PreconditionerBase<Number>>>(nonlinear_operator,
                                                    *preconditioner,
                                                    solver_control_l);

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
      std::cout << "solution:" << std::endl;
      for (unsigned int b = 0; b < src.n_blocks(); ++b)
        src.block(b).print(std::cout);
      std::cout << std::endl;

      nonlinear_operator.evaluate_nonlinear_residual(dst, src);

      std::cout << "residual:" << std::endl;
      for (unsigned int b = 0; b < dst.n_blocks(); ++b)
        dst.block(b).print(std::cout);
      std::cout << std::endl;
    };

    non_linear_solver->setup_jacobian =
      [&](const auto &current_u, const bool do_update_preconditioner) {
        (void)current_u;
        nonlinear_operator.do_update();

        if (do_update_preconditioner)
          preconditioner->do_update();
      };

    non_linear_solver->solve_with_jacobian = [&](const auto &src, auto &dst) {
      const unsigned int n_iterations = linear_solver->solve(dst, src);

      return n_iterations;
    };

    nonlinear_operator.update_state(solution);

    // setup manual constraints
    {
      auto &displ_constraints_indices =
        nonlinear_operator.get_zero_constraints_indices();

      // Apply DBC
      AffineConstraints<Number> constraints_dbc;
      DoFTools::make_zero_boundary_constraints(dof_handler, 1, constraints_dbc);
      constraints_dbc.close();

      for (unsigned int d = 0; d < dim; ++d)
        {
          displ_constraints_indices[d].clear();

          const auto &partitioner = matrix_free.get_vector_partitioner();
          for (const auto i : partitioner->locally_owned_range())
            if (constraints_dbc.is_constrained(i))
              displ_constraints_indices[d].emplace_back(
                partitioner->global_to_local(i));
        }
    }

    non_linear_solver->solve(solution);

    // Final configuration
    output_result(1);

    std::cout << "Solver type:           "
              << (matrix_based ? "matrix-based" : "matrix-free") << std::endl;
    std::cout << "Linear iterations:     " << statistics.n_linear_iterations()
              << std::endl;
    std::cout << "Non-linear iterations: " << statistics.n_newton_iterations()
              << std::endl;
    std::cout << "Residual evaluations:  "
              << statistics.n_residual_evaluations() << std::endl;
  }
};

int
main(int argc, char **argv)
{
  bool matrix_based = false;

  if (argc >= 2 && std::string(argv[1]) == "--matrix-based")
    matrix_based = true;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim         = 2;
  constexpr int fe_degree   = 1;
  constexpr int n_points_1D = fe_degree + 1;

  Test<dim, fe_degree, n_points_1D, double, VectorizedArray<double, 4>> runner;
  runner.run(matrix_based);
}
