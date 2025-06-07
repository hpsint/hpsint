// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------


// GP based KKS model with 2 phases as derived by Marco Seiz

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <chrono>
#include <fstream>
#include <unordered_map>

using namespace dealii;

template <int dim, int component>
class InitialValues : public Function<dim>
{};

template <int dim>
class InitialValues<dim, 0> : public Function<dim>
{
private:
  Point<dim> center{0.5, 0.5};

  std::function<double(double)> phifunc;

public:
  InitialValues(double W)
    : Function<dim>(1)
  {
    phifunc = [=](double x) { return 0.5 * (1.0 + std::tanh(x / W)); };
  }

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    const auto dist = center.distance(p);
    return phifunc(0.25 - dist);
  }
};

template <int dim>
class InitialValues<dim, 1> : public Function<dim>
{
private:
  InitialValues<dim, 0> phi0;

public:
  InitialValues(double W)
    : Function<dim>(1)
    , phi0(W)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    return 1 - phi0.value(p, 0);
  }
};

template <int dim>
class InitialValues<dim, 2> : public Function<dim>
{
private:
  InitialValues<dim, 0> phi0;
  InitialValues<dim, 1> phi1;

  std::array<double, 2> c_0;

public:
  InitialValues(double W, std::array<double, 2> c_0)
    : Function<dim>(1)
    , phi0(W)
    , phi1(W)
    , c_0{c_0}
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int) const override
  {
    // Smooth interface
    // return phi0.value(p, 0) * c_0[0] + phi1.value(p, 0) * c_0[1];

    // Sharp interface for c
    return (phi0.value(p, 0) > phi1.value(p, 0)) ? c_0[0] : c_0[1];
  }
};



template <int dim,
          int degree,
          int n_points_1D,
          int n_components,
          typename Number,
          typename VectorizedArrayType>
class MassMatrix
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    FEEvaluation<dim,
                 degree,
                 n_points_1D,
                 n_components,
                 Number,
                 VectorizedArrayType>
      phi(matrix_free);

    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, auto &range) {
        for (auto cell = range.first; cell < range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(phi.get_value(q), q);
            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      dst,
      src,
      true);
  }

  void
  initialize_dof_vector(VectorType &dst) const
  {
    matrix_free.initialize_dof_vector(dst);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
};



template <int dim,
          int fe_degree,
          int n_points_1D              = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  void
  run(const TimeStepping::runge_kutta_method method,
      const unsigned int                     n_refinements,
      const unsigned int                     n_time_steps,
      const unsigned int                     n_time_steps_output)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // Debug output
    constexpr bool print = false;

    // Dimensionality
    constexpr unsigned int n_comp = 3;
    constexpr unsigned int n_phi  = n_comp - 1;

    const std::unordered_map<unsigned int, std::string> labels = {{0, "phi0"},
                                                                  {1, "phi1"},
                                                                  {2, "c"}};

    // geometry and mesh
    const double       size           = 1.0;
    const unsigned int n_subdivisions = 1;

    const auto getW_if = [](const Number Wfac, const Number dx) {
      return Wfac * dx / std::log(99.0);
    };

    // Physical parameters
    const Number dx    = size / std::pow(2, n_refinements); // grid spacing
    const Number gamma = 1;
    const Number M     = 1;
    const Number dtfac = 0.98;
    const Number W     = getW_if(6, dx);
    const Number A     = 3 * W * gamma / 2;
    const Number B     = 6 * gamma / W;
    const Number L     = 2 * M / (3 * W);
    const Number D     = 1;

    // Key expressions
    const std::array<Number, n_phi> k    = {{500.0, 500.0}};
    const std::array<Number, n_phi> c_0  = {{0.02, 0.98}};
    const unsigned int              vidx = 0;
    const unsigned int              sidx = 1;
    const std::array<Number, n_phi> linf = {{1.0, -1.0}};

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_cube(tria, n_subdivisions, 0, size);
    tria.refine_global(n_refinements);

    const auto create_fe = []() {
      if constexpr (fe_degree == 0)
        return FE_DGQ<dim>(fe_degree);
      else
        return FE_Q<dim>(fe_degree);
    };

    auto            fe = create_fe();
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    MappingQ<dim> mapping(1);

    QGauss<1> quad(n_points_1D);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

    VectorType solution(n_comp), rhs(n_comp);

    for (unsigned int c = 0; c < n_comp; ++c)
      {
        matrix_free.initialize_dof_vector(solution.block(c));
        matrix_free.initialize_dof_vector(rhs.block(c));
      }

    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 0>(W),
                             solution.block(0));
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 1>(W),
                             solution.block(1));
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialValues<dim, 2>(W, c_0),
                             solution.block(2));

    const auto output_result =
      [&](const VectorType &vec, const double t, const unsigned int step) {
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        DataOut<dim> data_out;
        data_out.set_flags(flags);
        data_out.attach_dof_handler(dof_handler);

        for (unsigned int c = 0; c < n_comp; ++c)
          data_out.add_data_vector(vec.block(c),
                                   labels.at(c),
                                   DataOut_DoFData<dim, dim>::type_dof_data);

        vec.update_ghost_values();
        data_out.build_patches(mapping, fe_degree);

        static unsigned int counter = 0;

        pcout << "outputing at step = " << step << ", t = " << t << std::endl;

        std::string output = "solution." + std::to_string(counter++) + ".vtu";
        data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
      };

    FEEvaluation<dim,
                 fe_degree,
                 n_points_1D,
                 n_comp,
                 Number,
                 VectorizedArrayType>
      eval(matrix_free);

    MassMatrix<dim, fe_degree, n_points_1D, n_comp, Number, VectorizedArrayType>
      mass_matrix(matrix_free);

    // Timestep estimates
    const Number dt_phi =
      3 * W * W * dx * dx * gamma /
      (M * (2 * 12 * W * W * gamma * 2 + 0.96 * dx * dx * gamma * k[0] +
            12 * dx * dx * gamma * gamma));

    const Number dt_c = dx * dx / (2 * 4 * D * (1 + std::abs(c_0[0] - c_0[1])));
    const Number dt   = dtfac * std::min(dt_c, dt_phi);

    // Params
    pcout << "dx = " << dx << std::endl;
    pcout << "W  = " << W << std::endl;
    pcout << "A  = " << A << std::endl;
    pcout << "B  = " << B << std::endl;
    pcout << "L  = " << L << std::endl;
    pcout << "D  = " << D << std::endl;
    pcout << "dt = " << dt << std::endl;
    pcout << std::endl;

    // Some helper lambdas for the calculations
    const auto calc_ca = [&](const VectorizedArrayType                    &c,
                             const std::array<VectorizedArrayType, n_phi> &phi,
                             const unsigned int                            a) {
      const auto idx    = static_cast<unsigned>(a == 0);
      const auto prefac = linf[a];

      const auto nom = (c * k[idx] + prefac * (c_0[vidx] * k[vidx] * phi[idx] -
                                               c_0[sidx] * k[sidx] * phi[idx]));

      const auto denom = (k[0] * phi[1] + k[1] * phi[0]);

      return nom / denom;
    };

    const auto calc_grad_ca =
      [&](
        const VectorizedArrayType                                    &c,
        const Tensor<1, dim, VectorizedArrayType>                    &c_grad,
        const std::array<VectorizedArrayType, n_phi>                 &phi,
        const std::array<Tensor<1, dim, VectorizedArrayType>, n_phi> &phi_grad,
        const unsigned int                                            a) {
        const auto idx    = static_cast<unsigned>(a == 0);
        const auto prefac = linf[a];

        const auto coeff = prefac * (c_0[vidx] * k[vidx] - c_0[sidx] * k[sidx]);

        const auto nom        = c * k[idx] + coeff * phi[idx];
        const auto nom_grad   = c_grad * k[idx] + coeff * phi_grad[idx];
        const auto denom      = k[0] * phi[1] + k[1] * phi[0];
        const auto denom_grad = k[0] * phi_grad[1] + k[1] * phi_grad[0];

        return (nom_grad * denom - nom * denom_grad) / (denom * denom);
      };

    const auto calc_ga =
      [&](const VectorizedArrayType &ca, const Number ki, const Number c_0i) {
        return 0.5 * ki * std::pow(ca - c_0i, 2.);
      };

    const auto calc_mu = [&](const VectorizedArrayType &ca,
                             const Number               ki,
                             const Number c_0i) { return ki * (ca - c_0i); };

    // Some postprocessing of scalar quantities
    const auto check_sums = [&](const VectorType &vec, const double t) {
      std::vector<Number> sums(n_comp, 0.0);
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &, const auto &src, auto cells) {
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            {
              eval.reinit(cell);
              eval.gather_evaluate(src, EvaluationFlags::values);

              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                eval.submit_value(eval.get_value(q), q);

              const auto vals = eval.integrate_value();

              for (unsigned int c = 0; c < n_comp; ++c)
                sums[c] +=
                  std::accumulate(vals[c].begin(), vals[c].end(), Number(0));
            }
        },
        rhs,
        vec,
        false);

      for (unsigned int c = 0; c < n_comp; ++c)
        sums[c] = Utilities::MPI::sum<Number>(sums[c], MPI_COMM_WORLD);

      pcout << "sums at t = " << t << ":" << std::endl;
      for (unsigned int c = 0; c < n_comp; ++c)
        pcout << "  " << labels.at(c) << " = " << sums[c] << std::endl;
      pcout << std::endl;
    };

    // Initial output
    check_sums(solution, 0.0);
    output_result(solution, 0.0, 0);

    // RHS evaluation for the time-stepping
    const auto evaluate_rhs = [&](const double t, const VectorType &y) {
      VectorType incr(n_comp);
      for (unsigned int c = 0; c < n_comp; ++c)
        matrix_free.initialize_dof_vector(incr.block(c));

      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, auto cells) {
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            {
              eval.reinit(cell);
              eval.gather_evaluate(src,
                                   EvaluationFlags::values |
                                     EvaluationFlags::gradients);
              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                {
                  const auto value    = eval.get_value(q);
                  const auto gradient = eval.get_gradient(q);

                  // Field values
                  std::array<VectorizedArrayType, n_phi> phi = {
                    {value[0], value[1]}};
                  const auto c = value[2];

                  // Field gradients
                  const auto phi_grad =
                    std::array<Tensor<1, dim, VectorizedArrayType>, n_phi>{
                      {gradient[0], gradient[1]}};
                  const auto c_grad = gradient[2];

                  // Some auxiliary variables
                  const auto ca_0 = calc_ca(c, phi, 0);
                  const auto ca_1 = calc_ca(c, phi, 1);
                  const auto grad_ca_0 =
                    calc_grad_ca(c, c_grad, phi, phi_grad, 0);
                  const auto grad_ca_1 =
                    calc_grad_ca(c, c_grad, phi, phi_grad, 1);

                  // Chemical potential
                  const auto mu_0 = calc_mu(ca_0, k[0], c_0[0]);
                  const auto mu_1 = calc_mu(ca_1, k[1], c_0[1]);

                  const auto ga_0 = calc_ga(ca_0, k[0], c_0[0]);
                  const auto ga_1 = calc_ga(ca_1, k[1], c_0[1]);

                  // Number of active phases
                  const VectorizedArrayType zeroes(0.0), ones(1.0);

                  const auto nsz =
                    compare_and_apply_mask<SIMDComparison::greater_than>(
                      phi[0], zeroes, ones, zeroes) +
                    compare_and_apply_mask<SIMDComparison::greater_than>(
                      phi[1], zeroes, ones, zeroes);

                  auto vvars0 =
                    2. * B * phi[0] * std::pow(phi[1], 2.) + ga_0 - mu_0 * ca_0;
                  auto vvars1 =
                    2. * B * phi[1] * std::pow(phi[0], 2.) + ga_1 - mu_1 * ca_1;

                  Tensor<1, n_comp, VectorizedArrayType> value_result;
                  value_result[0] = -L / nsz * (vvars0 - vvars1);
                  value_result[1] = -L / nsz * (vvars1 - vvars0);
                  value_result[2] = 0.;

                  Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                    gradient_result;

                  gradient_result[0] =
                    L / nsz * A * (phi_grad[1] - phi_grad[0]);
                  gradient_result[1] =
                    L / nsz * A * (phi_grad[0] - phi_grad[1]);
                  gradient_result[2] =
                    -D * (phi[0] * grad_ca_0 + phi[1] * grad_ca_1);

                  eval.submit_value(value_result, q);
                  eval.submit_gradient(gradient_result, q);
                }
              eval.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     dst);
            }
        },
        rhs,
        y,
        true);

      {
        ReductionControl     reduction_control;
        SolverCG<VectorType> solver(reduction_control);
        solver.solve(mass_matrix, incr, rhs, PreconditionIdentity());

        if constexpr (print)
          pcout << "# of linear iterations at t = " << t << ": "
                << reduction_control.last_step() << std::endl;
      }

      return incr;
    };

    // time loop: variables are phi_0, phi_1, c
    const double initial_time = 0.0;
    const double final_time   = n_time_steps * dt;

    auto start = std::chrono::high_resolution_clock::now();

    TimeStepping::ExplicitRungeKutta<VectorType> explicit_runge_kutta(method);

    DiscreteTime time(initial_time, final_time, dt);
    while (time.is_at_end() == false)
      {
        explicit_runge_kutta.evolve_one_time_step(
          [&](const double t, const VectorType &y) {
            return evaluate_rhs(t, y);
          },
          time.get_current_time(),
          time.get_next_step_size(),
          solution);

        time.advance_time();

        if (time.get_step_number() % n_time_steps_output == 0 ||
            time.is_at_end())
          output_result(solution,
                        time.get_current_time(),
                        time.get_step_number());
      }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(end - start);

    pcout << std::endl;

    check_sums(solution, time.get_current_time());

    pcout << "Execution time: " << elapsed.count() << "s\n";
  }
};

char *
get_cmd_option(char **begin, char **end, const std::string &option)
{
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
    return *itr;

  return nullptr;
}

bool
cmd_option_exists(char **begin, char **end, const std::string &option)
{
  return std::find(begin, end, option) != end;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Test<2, 1>                       runner;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  const auto print_methods = [&]() {
    pcout << "Available methods: " << std::endl;
    pcout << "  forward_euler - 1st order forward Euler" << std::endl;
    pcout << "  rk3           - 3rd order Runge-Kutta" << std::endl;
    pcout << "  rk3_ssp       - 3rd SSP Runge-Kutta" << std::endl;
    pcout << "  rk4           - 4th order Runge-Kutta" << std::endl;
    pcout << "  ls_rk3_st3    - low-storage 3 stages 3rd order Runge-Kutta"
          << std::endl;
    pcout << "  ls_rk4_st5    - low-storage 5 stages 4th order Runge-Kutta"
          << std::endl;
    pcout << "  ls_rk4_st7    - low-storage 7 stages 4rd order Runge-Kutta"
          << std::endl;
    pcout << "  ls_rk5_st9    - low-storage 9 stages 5rd order Runge-Kutta"
          << std::endl;
  };

  if (cmd_option_exists(argv, argv + argc, "-h"))
    {
      pcout << "Options: " << std::endl;
      pcout << "  -h: print this help message" << std::endl;
      pcout
        << "  -m <method>: specify the time-stepping method (default: forward_euler)"
        << std::endl;
      print_methods();
      return 0;
    }

  char             *method_name = get_cmd_option(argv, argv + argc, "-m");
  const std::string method_str  = method_name ? method_name : "forward_euler";

  const std::unordered_map<std::string, TimeStepping::runge_kutta_method>
    method_map = {{"forward_euler", TimeStepping::FORWARD_EULER},
                  {"rk3", TimeStepping::RK_THIRD_ORDER},
                  {"rk3_ssp", TimeStepping::SSP_THIRD_ORDER},
                  {"rk4", TimeStepping::RK_CLASSIC_FOURTH_ORDER},
                  {"ls_rk3_st3", TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3},
                  {"ls_rk4_st5", TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4},
                  {"ls_rk4_st7", TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4},
                  {"ls_rk5_st9", TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5}};

  if (method_map.find(method_str) == method_map.end())
    {
      pcout << "Unknown method: " << method_str << std::endl;
      pcout << "Available methods: " << std::endl;
      for (const auto &method : method_map)
        pcout << "  " << method.first << std::endl;
      return 1;
    }
  else
    {
      pcout << "Using method: " << method_str << std::endl;
    }

  const auto method = method_map.at(method_str);

  // Number of refinements
  char              *n_refs_char = get_cmd_option(argv, argv + argc, "-r");
  const unsigned int n_refs =
    n_refs_char ? std::stoi(n_refs_char) : 6; // default to 6 refinements

  // Number of steps
  char              *n_steps_char = get_cmd_option(argv, argv + argc, "-s");
  const unsigned int n_steps =
    n_steps_char ? std::stoi(n_steps_char) : 1000; // default to 1000 steps

  // Number of steps to generate output
  char              *n_output_char = get_cmd_option(argv, argv + argc, "-o");
  const unsigned int n_steps_output =
    n_output_char ? std::stoi(n_output_char) : 100; // default to 100 steps

  pcout << std::endl;
  pcout << "Number of refinements: " << n_refs << std::endl;
  pcout << "Number of steps:       " << n_steps << std::endl;
  pcout << "Output at every:       " << n_steps_output << std::endl;
  pcout << std::endl;

  runner.run(method, n_refs, n_steps, n_steps_output);

  return 0;
}