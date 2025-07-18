#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/initial_values_debug.h>
#include <pf-applications/sintering/operator_advection.h>
#include <pf-applications/sintering/tools.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/matrix_free/tools.h>
#include <pf-applications/structural/material.h>
#include <pf-applications/time_integration/solution_history.h>
#include <pf-applications/time_integration/time_integrators.h>
#include <pf-applications/time_integration/time_schemes.h>

#include <iostream>

namespace Test
{
  using namespace dealii;
  using namespace Sintering;
  using namespace TimeIntegration;

  template <int dim,
            typename Number,
            typename VectorType,
            typename VectorizedArrayType,
            typename NonLinearOperator,
            typename FreeEnergy>
  struct SinteringModel
  {
  public:
    SinteringModel(const bool enable_rbm)
      : fe_degree(1)
      , tria(MPI_COMM_WORLD)
      , fe(fe_degree)
      , mapping(1)
      , quad(fe_degree + 1)
      , dof_handler(tria)
      , free_energy(/*A=*/1.,
                    /*B=*/1.)
      , sintering_data(
          /*kappa_c=*/0.1,
          /*kappa_p=*/0.1,
          std::make_shared<ProviderAbstract>(1e-1, 1e-8, 1e1, 1e0, 1e1),
          TimeIntegratorData<Number>(std::make_unique<BDF1Scheme<Number>>(),
                                     /*dt=*/1e-5))
      , solution_history(sintering_data.time_data.get_order() + 1)
      , advection_mechanism(enable_rbm,
                            /*mt=*/1.0,
                            /*mr=*/1.0)
    {
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

      create_mesh_from_divisions(tria,
                                 bottom_left,
                                 top_right,
                                 subdivisions,
                                 periodic,
                                 n_refinements,
                                 print_stats);

      // setup DoFHandlers
      dof_handler.distribute_dofs(fe);
      constraints.close();

      const unsigned int n_sinter_components = 4;

      sintering_data.set_n_components(n_sinter_components);

      typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags =
        update_values | update_gradients | update_quadrature_points;
      additional_data.allow_ghosted_vectors_in_loops    = false;
      additional_data.overlap_communication_computation = false;

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

      GrainTracker::GrainRepresentation grain_representation =
        GrainTracker::GrainRepresentation::spherical;

      grain_tracker = std::make_unique<GrainTracker::Tracker<dim, Number>>(
        dof_handler,
        tria,
        greedy_init,
        allow_new_grains,
        fast_reassignment,
        max_order_parameters_num,
        grain_representation,
        threshold_lower,
        threshold_upper,
        buffer_distance_ratio,
        buffer_distance_fixed,
        op_offset);

      // set initial condition
      InitialValuesDebug<dim> initial_solution;

      // External loading - not used at the moment
      auto body_force = [](const Point<dim, VectorizedArrayType> &p) {
        (void)p;

        Tensor<1, dim, VectorizedArrayType> value_result;
        value_result[0] = -0.5;
        value_result[1] = -0.2;
        if constexpr (dim == 3)
          value_result[2] = -0.7;

        return value_result;
      };

      const bool matrix_based                               = true;
      const bool use_tensorial_mobility_gradient_on_the_fly = false;

      // Elastic material properties - not used at the moment
      const double                        E     = 1;
      const double                        nu    = 0.25;
      const double                        c_min = 0.1;
      const Structural::MaterialPlaneType plane_type =
        Structural::MaterialPlaneType::plane_strain;

      // Create nonlinear_operator inplace since we already have sufficient
      // memory for it. Note that "delete" is not required to be explicitly
      // called, the dtor of SinteringModel deletes nonlinear_operator. This
      // code also extensively relies on RVO, since NonLinearOperator is neither
      // copy nor move assignable due to const members. Luckily, RVO was made
      // mandatory in C++17 for the way how create() is implemented (e.g., NRVO
      // will not work here).
      new (&nonlinear_operator) NonLinearOperator(
        NonLinearOperator::create(matrix_free,
                                  constraints,
                                  free_energy,
                                  sintering_data,
                                  solution_history,
                                  advection_mechanism,
                                  matrix_based,
                                  use_tensorial_mobility_gradient_on_the_fly,
                                  E,
                                  nu,
                                  plane_type,
                                  c_min,
                                  body_force));

      std::function<void(VectorType &)> f_init = [this](VectorType &v) {
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

      const double k           = 25;
      const double cgb         = 0;
      const double ceq         = 1.0;
      const double smoothening = 0;

      advection_operator =
        std::make_unique<AdvectionOperator<dim, Number, VectorizedArrayType>>(
          k,
          cgb,
          ceq,
          smoothening,
          matrix_free,
          constraints,
          sintering_data,
          *grain_tracker,
          advection_mechanism);

      grain_tracker->initial_setup(solution, sintering_data.n_grains());

      const bool save_all_blocks = true;
      sintering_data.fill_quadrature_point_values(matrix_free,
                                                  solution,
                                                  enable_rbm,
                                                  save_all_blocks);
    }

    ~SinteringModel()
    {}

    auto &
    get_nonlinear_operator()
    {
      return nonlinear_operator;
    }

    auto &
    get_advection_operator()
    {
      return *advection_operator;
    }

    auto &
    get_grain_tracker()
    {
      return *grain_tracker;
    }

    auto &
    get_advection_mechanism()
    {
      return advection_mechanism;
    }

    auto &
    get_dof_handler()
    {
      return dof_handler;
    }

    auto &
    get_solution()
    {
      return solution_history.get_current_solution();
    }

  private:
    const unsigned int fe_degree;

    parallel::distributed::Triangulation<dim>            tria;
    FE_Q<dim>                                            fe;
    MappingQ<dim>                                        mapping;
    QGauss<dim>                                          quad;
    DoFHandler<dim>                                      dof_handler;
    AffineConstraints<Number>                            constraints;
    FreeEnergy                                           free_energy;
    SinteringOperatorData<dim, VectorizedArrayType>      sintering_data;
    SolutionHistory<VectorType>                          solution_history;
    MatrixFree<dim, Number, VectorizedArrayType>         matrix_free;
    AdvectionMechanism<dim, Number, VectorizedArrayType> advection_mechanism;

    std::unique_ptr<GrainTracker::Tracker<dim, Number>> grain_tracker;
    std::unique_ptr<AdvectionOperator<dim, Number, VectorizedArrayType>>
      advection_operator;

    // This is to avoid default initialization
    union
    {
      NonLinearOperator nonlinear_operator;
    };
  };

} // namespace Test