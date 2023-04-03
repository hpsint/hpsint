#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/creator.h>
#include <pf-applications/sintering/initial_values_debug.h>
#include <pf-applications/sintering/operator_advection.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/matrix_free/tools.h>

#include <iostream>

namespace Test
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename NonLinearOperator>
  void
  check_tangent(const bool   enable_rbm,
                const double tol_abs = 1e-3,
                const double tol_rel = 1e-6)
  {
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    const unsigned int fe_degree = 1;

    const bool is_zero_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    ConditionalOStream                        pcout(std::cout, is_zero_rank);
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    FE_Q<dim>                                 fe(fe_degree);
    MappingQ<dim>                             mapping(1);
    QGauss<dim>                               quad(fe_degree + 1);
    DoFHandler<dim>                           dof_handler(tria);
    AffineConstraints<Number>                 constraints;

    Point<dim> bottom_left;
    Point<dim> top_right;
    top_right[0] = 2.0;
    for (unsigned int d = 1; d < dim; ++d)
      top_right[d] = 1.0;

    std::vector<unsigned int> subdivisions(dim, 1);
    subdivisions[0] = 2;

    // Mesh settings
    const bool         periodic      = false;
    const unsigned int n_refinements = 0;
    const bool         print_stats   = false;

    create_mesh(tria,
                bottom_left,
                top_right,
                subdivisions,
                periodic,
                n_refinements,
                print_stats);

    // setup DoFHandlers
    dof_handler.distribute_dofs(fe);
    constraints.close();

    // Additional objects
    const double       A                      = 1.;
    const double       B                      = 1.;
    const double       kappa_c                = 0.1;
    const double       kappa_p                = 0.1;
    const unsigned int time_integration_order = 1;
    const unsigned int n_sinter_components    = 4;

    std::shared_ptr<MobilityProvider> mobility_provider =
      std::make_shared<ProviderAbstract>(1e-1, 1e-8, 1e1, 1e0, 1e1);

    SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
      A, B, kappa_c, kappa_p, mobility_provider, time_integration_order);

    sintering_data.set_n_components(n_sinter_components);

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags =
      update_values | update_gradients | update_quadrature_points;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints, quad, additional_data);

    // Grain tracker settings
    const double       threshold_lower          = 1e-15;
    const double       threshold_upper          = 1.01;
    const double       buffer_distance_ratio    = 0.05;
    const double       buffer_distance_fixed    = 0.0;
    const bool         allow_new_grains         = false;
    const bool         greedy_init              = false;
    const bool         fast_reassignment        = false;
    const unsigned int op_offset                = 2;
    const unsigned int max_order_parameters_num = 2;

    GrainTracker::Tracker<dim, Number> grain_tracker(dof_handler,
                                                     tria,
                                                     greedy_init,
                                                     allow_new_grains,
                                                     fast_reassignment,
                                                     max_order_parameters_num,
                                                     threshold_lower,
                                                     threshold_upper,
                                                     buffer_distance_ratio,
                                                     buffer_distance_fixed,
                                                     op_offset);

    const double mt = 1.;
    const double mr = 1.;

    AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism(
      enable_rbm, mt, mr);

    // set initial condition
    InitialValuesDebug<dim> initial_solution;

    TimeIntegration::SolutionHistory<VectorType> solution_history(2);

    // External loading
    auto body_force_x = [](const Point<dim, VectorizedArrayType> &p) {
      (void)p;

      Tensor<1, dim, VectorizedArrayType> value_result;
      value_result[0] = -0.5;

      return value_result;
    };

    const bool matrix_based                               = true;
    const bool use_tensorial_mobility_gradient_on_the_fly = false;

    // Elastic material properties
    const double                        E  = 1;
    const double                        nu = 0.25;
    const Structural::MaterialPlaneType plane_type =
      Structural::MaterialPlaneType::plane_strain;

    auto nonlinear_operator = create_sintering_operator<dim,
                                                        Number,
                                                        VectorizedArrayType,
                                                        VectorType,
                                                        NonLinearOperator>(
      matrix_free,
      constraints,
      sintering_data,
      solution_history,
      advection_mechanism,
      matrix_based,
      use_tensorial_mobility_gradient_on_the_fly,
      E,
      nu,
      plane_type,
      body_force_x);

    std::function<void(VectorType &)> f_init =
      [&nonlinear_operator](VectorType &v) {
        nonlinear_operator.initialize_dof_vector(v);
      };

    solution_history.apply(f_init);
    solution_history.zero_out_ghost_values();

    VectorType &solution = solution_history.get_current_solution();

    for (unsigned int c = 0; c < initial_solution.n_components(); ++c)
      {
        initial_solution.set_component(c);

        VectorTools::interpolate(mapping,
                                 dof_handler,
                                 initial_solution,
                                 solution.block(c));

        constraints.distribute(solution.block(c));
      }

    solution_history.set_recent_old_solution(solution);
    std::vector<Number> dts(time_integration_order);
    dts[0] = 1e-5;
    sintering_data.time_data.set_all_dt(dts);

    double k   = 100;
    double cgb = 0.1;
    double ceq = 1.0;

    AdvectionOperator<dim, Number, VectorizedArrayType> advection_operator(
      k, cgb, ceq, matrix_free, constraints, sintering_data, grain_tracker);

    grain_tracker.initial_setup(solution, sintering_data.n_grains());

    advection_operator.evaluate_forces(solution, advection_mechanism);

    const bool save_all_blocks = true;
    sintering_data.fill_quadrature_point_values(matrix_free,
                                                solution,
                                                enable_rbm,
                                                save_all_blocks);

    const double epsilon   = 1e-7;
    const double tolerance = 1e-12;

    // Nonlinear sintering operator
    VectorType residual;
    nonlinear_operator.initialize_dof_vector(residual);

    nonlinear_operator.evaluate_nonlinear_residual(residual, solution);

    const unsigned int n_blocks = solution.n_blocks();

    const unsigned int n_dofs = dof_handler.n_dofs() * n_blocks;

    FullMatrix<Number> tangent_analytic(n_dofs, n_dofs);
    nonlinear_operator.initialize_system_matrix(false);
    tangent_analytic.copy_from(nonlinear_operator.get_system_matrix());

    const auto         residual0(residual);
    FullMatrix<Number> tangent_numeric(n_dofs, n_dofs);

    const auto locally_owned_dofs = dof_handler.locally_owned_dofs();

    for (unsigned int b = 0; b < n_blocks; ++b)
      for (unsigned int i = 0; i < solution.block(b).size(); ++i)
        {
          auto residual1(residual);
          residual1 = 0;

          if (locally_owned_dofs.is_element(i))
            solution.block(b)[i] += epsilon;

          advection_operator.evaluate_forces(solution, advection_mechanism);

          nonlinear_operator.evaluate_nonlinear_residual(residual1, solution);

          if (locally_owned_dofs.is_element(i))
            solution.block(b)[i] -= epsilon;

          for (unsigned int b_ = 0; b_ < n_blocks; ++b_)
            for (unsigned int i_ = 0; i_ < solution.block(b).size(); ++i_)
              if (locally_owned_dofs.is_element(i_))
                {
                  if (nonlinear_operator.get_sparsity_pattern().exists(
                        b_ + i_ * n_blocks, b + i * n_blocks))
                    {
                      const Number value =
                        (residual1.block(b_)[i_] - residual0.block(b_)[i_]) /
                        epsilon;

                      if (std::abs(value) > tolerance)
                        tangent_numeric[b_ + i_ * n_blocks][b + i * n_blocks] =
                          value;

                      else if ((b == b_) && (i == i_))
                        tangent_numeric[b_ + i_ * n_blocks][b + i * n_blocks] =
                          1.0;
                    }
                }
        }

    auto tangent_diff(tangent_analytic);
    tangent_diff.add(Number(-1.), tangent_numeric);

    const auto comm = dof_handler.get_communicator();

    const auto error_abs = tangent_diff.l1_norm();
    const auto error_rel = error_abs / tangent_analytic.l1_norm();

    const bool is_correct = error_abs < tol_abs && error_rel < tol_rel;
    const bool all_correct =
      (dealii::Utilities::MPI::sum<unsigned int>(is_correct, comm) ==
       Utilities::MPI::n_mpi_processes(comm));

    pcout << "Tangent is " << (all_correct ? "OK" : "ERROR")
          << " within tol_abs = " << tol_abs << " tol_rel = " << tol_rel
          << std::endl;

    if (!all_correct)
      {
        pcout << std::endl;
        grain_tracker.print_current_grains(pcout, true);
        pcout << std::endl;

        std::ostringstream ss;

        ss << std::endl;
        ss << "===== Output from rank "
           << Utilities::MPI::this_mpi_process(comm)
           << " (total = " << Utilities::MPI::n_mpi_processes(comm)
           << ") =====" << std::endl;

        ss << std::endl;
        ss << "diff L2 norm absolute = " << error_abs
           << " (tol_abs = " << tol_abs << ") - "
           << (error_abs < tol_abs ? "OK" : "ERROR") << std::endl;
        ss << "diff L2 norm relative = " << error_rel
           << " (tol_rel = " << tol_rel << ") - "
           << (error_rel < tol_rel ? "OK" : "ERROR") << std::endl;
        ss << std::endl;

        ss << std::endl << "Tangent analytic:" << std::endl;
        tangent_analytic.print_formatted(ss);

        ss << std::endl << "Tangent numeric:" << std::endl;
        tangent_numeric.print_formatted(ss);

        ss << std::endl << "Tangent diff:" << std::endl;
        tangent_diff.print_formatted(ss);

        ss << std::endl << "Solution:" << std::endl;
        for (unsigned int c = 0; c < solution.n_blocks(); ++c)
          {
            ss << "block " << c << ": ";
            solution.block(c).print(ss);
            ss << std::endl;
          }

        ss << std::endl << "Residual:" << std::endl;
        for (unsigned int c = 0; c < residual.n_blocks(); ++c)
          {
            ss << "block " << c << ": ";
            residual.block(c).print(ss);
            ss << std::endl;
          }

        auto all_prints = Utilities::MPI::gather(comm, ss.str());

        for (const auto &entry : all_prints)
          pcout << entry;
      }
  }
} // namespace Test