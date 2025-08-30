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
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/base/data.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grid/grid_tools.h>

#include <chrono>
#include <fstream>
#include <unordered_map>

#include "kks_helpers.h"

using namespace dealii;
using namespace KKS;
using namespace hpsint;

template <typename Number>
Number
phifunc(Number x_ref, Number factor = 1.0)
{
  return factor * 0.5 * (1.0 + std::tanh(x_ref));
}

template <typename Number>
Number
phifunc_lin(Number x, Number center, Number width)
{
  if (x <= center - width / 2.)
    return 0.;
  else if (x >= center + width / 2.)
    return 1.;
  else
    return (x - center + width / 2.) / width;
}

template <int dim>
class InitialValues : public Function<dim>
{
public:
  InitialValues()
    : Function<dim>(1)
  {}

  virtual unsigned int
  n_components() const = 0;

  void
  set_component(const unsigned int component)
  {
    AssertIndexRange(component, n_components());
    current_component = component;
  }

protected:
  unsigned int current_component = 0;
};

template <int dim>
class InitialValues1D : public InitialValues<dim>
{
public:
  InitialValues1D(bool                  smooth_in,
                  double                noise_in,
                  double                size,
                  double                W_in,
                  std::array<double, 2> c_0_in)
    : InitialValues<dim>()
    , smooth(smooth_in)
    , noise(noise_in)
    , length(size)
    , W(W_in)
    , c_0(c_0_in)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    if (this->current_component < n_components() - 1)
      return do_phi(p, this->current_component);
    else
      {
        const double phi0sq   = std::pow(do_phi(p, 0), 2);
        double       phisqsum = phi0sq;
        for (unsigned int i = 1; i < n_components() - 1; ++i)
          phisqsum += std::pow(do_phi(p, i), 2);

        const double phiv = phi0sq / phisqsum;

        return phiv * c_0[0] + (1. - phiv) * c_0[1];
      }
  }

  unsigned int
  n_components() const override
  {
    return 4; // phi0, phi1, phi2, c
  }

private:
  const bool   smooth;
  const double noise;
  const double length = 1;
  const double W;

  std::array<double, 2> c_0;

  double
  do_phi(const Point<dim> &p, const unsigned int component) const
  {
    const auto factor = 1. - noise;

    if (component == 0)
      {
        if (smooth)
          {
            return phifunc((p[0] - length / 2.) / W, factor);
          }
        else
          {
            return p[0] >= length / 2. ? factor : 0.0;
          }
      }
    else if (component == 1)
      {
        if (smooth)
          {
            return factor - phifunc((p[0] - length / 2.) / W, factor);
          }
        else
          {
            return p[0] < length / 2. ? factor : 0.0;
          }
      }
    else if (component == 2)
      {
        return noise;
      }
    else
      {
        AssertThrow(false, ExcMessage("Invalid component index"));
        return 0.0; // to avoid compiler warning
      }
  }
};

// Debug 1D DG0 as FDM
template <int dim>
class Debug1D : public InitialValues<dim>
{
public:
  Debug1D()
    : InitialValues<dim>()
  {}

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    const auto val = std::floor(p[0]);
    return this->current_component ? (4. - val) : val;
  }

  unsigned int
  n_components() const override
  {
    return 2;
  }
};

// Initial values square with a circle
template <int dim>
class InitialValuesCircle : public InitialValues<dim>
{
public:
  InitialValuesCircle(double                r_in,
                      bool                  smooth_in,
                      double                W_in,
                      std::array<double, 2> c_0_in)
    : InitialValues<dim>()
    , radius(r_in)
    , smooth_interface(smooth_in)
    , W(W_in)
    , c_0(c_0_in)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    if (this->current_component < n_components() - 1)
      return do_phi(p, this->current_component);
    else
      {
        const auto phi0 = do_phi(p, 0);
        const auto phi1 = do_phi(p, 1);
        if (smooth_interface)
          return phi0 * c_0[0] + phi1 * c_0[1];
        else
          return (phi0 > phi1) ? c_0[0] : c_0[1];
      }
  }

  unsigned int
  n_components() const override
  {
    return 3; // phi0, phi1, c
  }

private:
  Point<dim> center{0.5, 0.5};

  const double radius;
  const bool   smooth_interface;
  const double W;

  std::array<double, 2> c_0;

  double
  do_phi(const Point<dim> &p, const unsigned int component) const
  {
    const auto dist = center.distance(p);
    const auto ksi  = (radius - dist) / W;

    if (component == 0)
      {
        return phifunc(ksi);
      }
    else if (component == 1)
      {
        return 1. - phifunc(ksi);
      }
    else
      {
        AssertThrow(false, ExcMessage("Invalid component index"));
        return 0.0; // to avoid compiler warning
      }
  }
};

template <int dim>
class InitialValuesSquare : public InitialValues<dim>
{
public:
  InitialValuesSquare(double width_in)
    : InitialValues<dim>()
    , width(width_in)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    return do_phi(p, this->current_component);
  }

  unsigned int
  n_components() const override
  {
    return 3; // phi0, phi1, phi2
  }

private:
  double     width;
  Point<dim> center{0.5, 0.5};

  std::array<double, 3>
  do_all_phis(const Point<dim> &p) const
  {
    std::array<double, 3> phis;
    phis[0] = phifunc_lin(p[1], center[1], width);
    phis[1] = std::max(1. - phifunc_lin(p[0], center[0], width) -
                         phifunc_lin(p[1], center[1], width),
                       0.);
    phis[2] = std::max(phifunc_lin(p[0], center[0], width) -
                         phifunc_lin(p[1], center[1], width),
                       0.);

    auto phi_sum =
      std::accumulate(phis.begin(), phis.end(), 0.0, std::plus<double>());
    if (phi_sum == 0.)
      phi_sum = 1.;

    // Marco's normalization
    std::transform(phis.begin(),
                   phis.end(),
                   phis.begin(),
                   [phi_sum](double &val) { return val / phi_sum; });

    return phis;
  }

  // Has to be redefined as does not produce driving force with Gauss-Lobatto
  // integration rules
  double
  do_phi(const Point<dim> &p, const unsigned int component) const
  {
    if (component > 2)
      {
        AssertThrow(false, ExcMessage("Invalid component index"));
        return 0.0; // to avoid compiler warning
      }

    if (p == center)
      {
        if (component == 0)
          return 0.5;
        else
          return 0.25;
      }
    else
      {
        return do_all_phis(p)[component];
      }
  }
};

template <int dim>
class InitialValuesPore : public InitialValues<dim>
{
public:
  InitialValuesPore(double                size_x_in,
                    double                size_y_in,
                    double                radius_in,
                    double                W_in,
                    bool                  is_smooth_in,
                    std::array<double, 2> c_0_in)
    : InitialValues<dim>()
    , size_x(size_x_in)
    , size_y(size_y_in)
    , radius(radius_in)
    , W(W_in)
    , is_smooth(is_smooth_in)
    , center(0.5 * size_x, 0.5 * size_y)
    , c_0(c_0_in)
  {}

  double
  value(const Point<dim> &p, const unsigned int) const override
  {
    if (this->current_component < n_components() - 1)
      return do_phi(p, this->current_component);
    else
      {
        const auto phis  = do_all_phis(p);
        const auto hphi0 = hfunc2_0(phis);

        return hphi0 * c_0[0] + (1. - hphi0) * c_0[1];
      }
  }

  unsigned int
  n_components() const override
  {
    return 4; // phi0, phi1, phi2, c
  }

private:
  double     size_x;
  double     size_y;
  double     radius;
  double     W;
  bool       is_smooth;
  Point<dim> center;

  std::array<double, 2> c_0;

  double
  get_phase_field_value(const Point<dim> &p) const
  {
    const double x = p[0] - center[0]; // Centering X
    const double y = p[1] - center[1]; // Centering Y

    // Interface definition
    const double r         = radius;
    bool         in_top    = y > 0;
    bool         in_circle = (x * x + y * y <= r * r);
    bool         indicator = in_top && !in_circle;

    double phi;
    if (indicator)
      {
        double dist_plane  = y;
        double dist_circle = std::sqrt(x * x + y * y) - r;
        phi                = std::min(dist_plane, dist_circle);
      }
    else
      {
        if (y <= 0)
          {
            if (x < -r || x > r)
              phi = y;
            else
              phi = -std::sqrt(std::pow(std::abs(x) - r, 2) + y * y);
          }
        else
          {
            phi = -(r - std::sqrt(x * x + y * y));
          }
      }

    return phifunc(phi / W);
  }

  std::array<double, 3>
  do_all_phis(const Point<dim> &p) const
  {
    const auto ksi0 = (radius - center.distance(p)) / W;

    std::array<double, 3> phis;
    phis[0] = phifunc(ksi0);

    if (is_smooth)
      {
        phis[1] = get_phase_field_value(p);

        Point<dim> inverted;
        inverted[0] = p[0];
        inverted[1] = size_y - p[1];
        phis[2]     = get_phase_field_value(inverted);
      }
    else
      {
        const auto ksi1 = (p[1] - center[1]) / W;

        phis[1] = std::max(phifunc(ksi1) - phifunc(ksi0), 0.);
        phis[2] = std::max(1. - phifunc(ksi1) - phifunc(ksi0), 0.);
      }

    auto phi_sum =
      std::accumulate(phis.begin(), phis.end(), 0.0, std::plus<double>());
    if (phi_sum == 0.)
      phi_sum = 1.;

    /*
  const auto delta = 1. - phi_sum;
  const double eps = 1e-4;
  if (delta > eps)
    {
      if (p[1] > center[1])
        phis[1] += delta;
      else if (p[1] < center[1])
        phis[2] += delta;
      else
        {
          phis[1] += delta / 2.;
          phis[2] += delta / 2.;
        }
    }
        */

    // Marco's normalization
    std::transform(phis.begin(),
                   phis.end(),
                   phis.begin(),
                   [phi_sum](double &val) { return val / phi_sum; });

    return phis;
  }

  double
  do_phi(const Point<dim> &p, const unsigned int component) const
  {
    if (component > 2)
      {
        AssertThrow(false, ExcMessage("Invalid component index"));
        return 0.0; // to avoid compiler warning
      }

    return do_all_phis(p)[component];
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
  using VectorType   = LinearAlgebra::distributed::DynamicBlockVector<Number>;
  using value_type   = Number;
  using FEIntegrator = FEEvaluation<dim,
                                    degree,
                                    n_points_1D,
                                    n_components,
                                    Number,
                                    VectorizedArrayType>;

  MassMatrix(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    FEIntegrator phi(matrix_free);

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
  compute_diagonal(VectorType &diagonal) const
  {
    diagonal.reinit(n_components);

    for (unsigned int b = 0; b < diagonal.n_blocks(); ++b)
      matrix_free.initialize_dof_vector(diagonal.block(b));

    const std::function<void(FEIntegrator &)> evaluate_cell =
      [&](FEIntegrator &phi) {
        phi.evaluate(EvaluationFlags::EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi.get_value(q), q);

        phi.integrate(EvaluationFlags::EvaluationFlags::values);
      };

    MatrixFreeTools::compute_diagonal(matrix_free, diagonal, evaluate_cell);
  }

  void
  initialize_dof_vector(VectorType &dst) const
  {
    matrix_free.initialize_dof_vector(dst);
  }

private:
  const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
};

template <typename Operator>
class InverseDiagonalMatrix
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<
    typename Operator::value_type>;

  InverseDiagonalMatrix(const Operator &op)
  {
    op.compute_diagonal(diagonal_matrix.get_vector());

    for (unsigned int b = 0; b < diagonal_matrix.get_vector().n_blocks(); ++b)
      for (auto &i : diagonal_matrix.get_vector().block(b))
        i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    diagonal_matrix.vmult(dst, src);
  }

private:
  DiagonalMatrix<VectorType> diagonal_matrix;
};



template <int  dim,
          int  fe_degree,
          bool is_dg                   = false,
          int  n_points_1D             = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
class Test
{
public:
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  void
  run(const TimeStepping::runge_kutta_method method,
      const int                              n_refinements_additional,
      const unsigned int                     n_time_steps_custom,
      const unsigned int                     n_time_steps_output,
      const bool                             do_amr)
  {
    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // Debug output
    constexpr bool print       = true;
    constexpr bool print_debug = false;

    // Simulation cases

    // OK
    const std::string      case_name       = "1d";
    constexpr bool         do_couple_phi_c = true;
    constexpr unsigned int n_comp          = 3 + do_couple_phi_c;

    // const std::string      case_name = "circle";
    // constexpr unsigned int n_comp    = 3;
    // constexpr bool do_couple_phi_c   = true;

    // OK
    // const std::string      case_name       = "gg2d";
    // constexpr unsigned int n_comp          = 2;
    // constexpr bool         do_couple_phi_c = false;

    // const std::string      case_name       = "gs1d";
    // const std::string case_name = "gg1d";
    // const std::string      case_name       = "debug1d";
    // constexpr unsigned int n_comp          = 2;
    // constexpr bool         do_couple_phi_c = false;

    // const std::string      case_name       = "tp";
    // constexpr unsigned int n_comp          = 3;
    // constexpr bool         do_couple_phi_c = false;

    // const std::string      case_name       = "tp-ggpore";
    // constexpr unsigned int n_comp          = 4;
    // constexpr bool         do_couple_phi_c = true;

    // Dimensionality
    constexpr unsigned int n_phi =
      n_comp - static_cast<unsigned int>(do_couple_phi_c);

    // Labels
    std::unordered_map<unsigned int, std::string> labels;
    for (unsigned int i = 0; i < n_phi; ++i)
      labels[i] = "phi" + std::to_string(i);
    if (do_couple_phi_c)
      labels[n_phi] = "c";

    const auto getW_if = [](const Number Wfac, const Number dx) {
      return Wfac * dx / std::log(99.0);
    };

    // Energy properties
    constexpr std::array<Number, 2> k   = {{5000.0, 5000.0}};
    constexpr std::array<Number, 2> c_0 = {{0.02, 0.98}};

    // Physical parameters and geometry - case specific
    double                    dx, W, dtfac;
    unsigned int              n_refinements;
    std::vector<unsigned int> subdivisions(dim, 1);
    Point<dim>                p1, p2;

    // Initial values
    std::unique_ptr<InitialValues<dim>> initial_values;

    // Number of timesteps to run
    unsigned int n_time_steps_default = 0;

    if (case_name == "debug1d")
      {
        n_refinements = 0;

        const unsigned int nx   = 4;
        const Number       size = 4.0;

        dx = size / nx;

        dtfac = 0.5; // 0.98;
        W     = getW_if(6, dx);

        subdivisions = {nx, 1};
        p2[0]        = size;
        p2[1]        = dx;

        initial_values = std::make_unique<Debug1D<dim>>();

        n_time_steps_default = 1;
      }
    else if (case_name == "1d")
      {
        n_refinements = 0 + n_refinements_additional;

        const unsigned int nx   = 64;
        const Number       size = 1.0;

        dx = size / nx / std::pow(2, n_refinements);

        const Number dx_w = size / 64;

        /* The noise phase pinns at the end with nx = 64, whereas disappears
         * completely if nx = 128 */

        dtfac = 0.5; // 0.98;
        W     = getW_if(6, dx_w);

        subdivisions = {nx, 1};
        p2[0]        = size;
        p2[1]        = dx;

        // DEBUG to coincide fully with native FDM
        /*
        dx = 1.0 / 64;
        p1[0] += 0.5 * dx;
        p2[0] -= 0.5 * dx;
        n_refinements = 0;
        subdivisions  = {63, 1};
        */

        const double noise = 0.1; // noise in initial values

        initial_values =
          std::make_unique<InitialValues1D<dim>>(true, noise, size, W, c_0);

        n_time_steps_default = 1000;
      }
    else if (case_name == "gg1d")
      {
        n_refinements = 0 + n_refinements_additional;

        const unsigned int nx       = 64;
        const Number       size     = 1.0;
        const unsigned int nx_ratio = 1;

        dx = size / nx;

        dtfac = 0.5;
        W     = 1.2 * 0.0250346565398582;
        // The larger W, the larger the interface width

        subdivisions = {nx_ratio * nx, 1};
        p2[0]        = size;
        p2[1]        = dx;

        const double noise = 0.0;

        initial_values =
          std::make_unique<InitialValues1D<dim>>(true, noise, size, W, c_0);

        n_time_steps_default = 100;
      }
    else if (case_name == "gs1d")
      {
        n_refinements = 0 + n_refinements_additional;

        const unsigned int nx       = 64;
        const Number       size     = 1.0;
        const unsigned int nx_ratio = 1;

        dx = size / nx;

        const Number dx_w = size / 64;

        dtfac = 0.5;
        W     = getW_if(6, dx_w);

        subdivisions = {nx_ratio * nx, 1};
        p2[0]        = size;
        p2[1]        = dx;

        const double noise = 0.0;

        initial_values =
          std::make_unique<InitialValues1D<dim>>(false, noise, size, W, c_0);

        n_time_steps_default = 100;
      }
    else if (case_name == "gg2d")
      {
        n_refinements = 6 + n_refinements_additional;

        const Number size = 1.0;

        dx = size / std::pow(2, n_refinements); // grid spacing

        const unsigned int nx = 1;

        const bool smooth_interface = true;

        dtfac = 0.5;
        W     = 1.2 * 0.0250346565398582;

        subdivisions = {nx, nx};
        p2[0]        = size;
        p2[1]        = size;

        initial_values = std::make_unique<InitialValuesCircle<dim>>(
          0.35, smooth_interface, W, c_0);

        n_time_steps_default = 1000;
      }
    else if (case_name == "circle")
      {
        n_refinements = 6 + n_refinements_additional;

        const Number size = 1.0;
        const Number dx   = size / std::pow(2, n_refinements); // grid spacing

        const unsigned int nx = 1;

        const bool smooth_interface = false;

        dtfac = 0.98; // 0.98;
        W     = getW_if(6, dx);

        subdivisions = {nx, nx};
        p2[0]        = size;
        p2[1]        = size;

        initial_values = std::make_unique<InitialValuesCircle<dim>>(
          0.25, smooth_interface, W, c_0);

        n_time_steps_default = 100;
      }
    else if (case_name == "tp")
      {
        n_refinements = 6 + n_refinements_additional;

        const Number size = 1.0;

        dx = size / std::pow(2, n_refinements); // grid spacing

        const unsigned int nx = 1;

        dtfac = 0.5;
        W     = 1.2 * 0.0250346565398582;

        subdivisions = {nx, nx};
        p2[0]        = size;
        p2[1]        = size;

        const Number width = dx;

        initial_values = std::make_unique<InitialValuesSquare<dim>>(width);

        n_time_steps_default = 1000;
      }
    else if (case_name == "tp-ggpore")
      {
        n_refinements = 4 + n_refinements_additional;

        const Number       size_x = 0.1;
        const unsigned int nx     = 6;
        const unsigned int ny     = 5;
        const Number       size_y =
          (static_cast<Number>(ny) / static_cast<Number>(nx)) * size_x;

        dx = size_x / nx / std::pow(2, n_refinements);

        const Number dx_w = size_x / 96; // the reference case

        dtfac = 0.5;
        W     = getW_if(6, dx_w);

        subdivisions = {nx, ny};
        p2[0]        = size_x;
        p2[1]        = size_y;

        const bool   is_smooth = false;
        const Number radius    = 0.35 * size_x / 2.0;

        initial_values = std::make_unique<InitialValuesPore<dim>>(
          size_x, size_y, radius, W, is_smooth, c_0);

        n_time_steps_default = 1000;
      }
    else
      {
        AssertThrow(false, ExcMessage("Unknown case name"));
      }

    const unsigned int n_time_steps =
      n_time_steps_custom ? n_time_steps_custom : n_time_steps_default;

    // Material properties
    const Number gsv = 1;
    const Number gss = 0.5;

    Matrix2x2<Number>  gamma{{{{0, gsv}}, {{gsv, gss}}}};
    Matrix2x2<Number>  mobi{{{{0, 1.}}, {{1., 1.}}}};
    PropMatrix<Number> A(gamma, 3. * W * 0.5);
    PropMatrix<Number> B(gamma, 6. / W);
    PropMatrix<Number> L(mobi, 1. / 3. / W);

    // Mobility
    const Number D = 1;

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, p1, p2);

    const auto create_fe = [&]() {
      if constexpr (is_dg)
        return FE_DGQ<dim>(fe_degree);
      else
        return FE_Q<dim>(fe_degree);
    };

    const auto create_quad = [&]() {
      if constexpr (n_points_1D > 1)
        return QGaussLobatto<1>(n_points_1D);
      // return QGauss<1>(n_points_1D);
      else
        return QGauss<1>(n_points_1D);
    };

    auto fe   = create_fe();
    auto quad = create_quad();

    DoFHandler<dim> dof_handler(tria);

    MappingQ<dim> mapping(1);

    AffineConstraints<Number> constraint;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
    additional_data.mapping_update_flags_inner_faces =
      update_values | update_gradients | update_JxW_values |
      update_quadrature_points;

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    VectorType solution(n_comp), rhs(n_comp);

    const auto initialize = [&]() {
      dof_handler.distribute_dofs(fe);

      constraint.clear();
      constraint.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);

      constraint.close();

      if constexpr (is_dg && n_points_1D == 1)
        {
          MatrixFreeTools::categorize_by_boundary_ids(
            dof_handler.get_triangulation(), additional_data);
        }
      matrix_free.reinit(
        mapping, dof_handler, constraint, quad, additional_data);

      for (unsigned int c = 0; c < n_comp; ++c)
        {
          matrix_free.initialize_dof_vector(solution.block(c));
          matrix_free.initialize_dof_vector(rhs.block(c));
        }
    };

    const auto execute_coarsening_and_refinement = [&]() {
      // and limit the number of levels
      const unsigned int max_allowed_level = n_refinements;
      const unsigned int min_allowed_level = 0;

      std::function<void(VectorType &)> after_amr = [&](VectorType &) {
        initialize();
      };

      const unsigned int block_estimate_start = 0;
      const unsigned int block_estimate_end   = n_phi;

      const Number interface_threshold = 0.01;
      const Number interface_val_min   = 0.0 + interface_threshold;
      const Number interface_val_max   = 1.0 - interface_threshold;

      const double top_fraction_of_cells    = 0.3;
      const double bottom_fraction_of_cells = 0.1;

      coarsen_and_refine_mesh(solution,
                              tria,
                              dof_handler,
                              constraint,
                              Quadrature<dim - 1>(quad),
                              top_fraction_of_cells,
                              bottom_fraction_of_cells,
                              min_allowed_level,
                              max_allowed_level,
                              after_amr,
                              interface_val_min,
                              interface_val_max,
                              block_estimate_start,
                              block_estimate_end);
    };

    auto initialize_solution = [&]() {
      for (unsigned int c = 0; c < n_comp; ++c)
        {
          initial_values->set_component(c);
          VectorTools::interpolate(mapping,
                                   dof_handler,
                                   *initial_values,
                                   solution.block(c));
          constraint.distribute(solution.block(c));
          solution.block(c).zero_out_ghost_values();
        }
    };

    pcout << "Number of total refinements: " << n_refinements << std::endl;

    if (do_amr)
      {
        initialize();
        initialize_solution();

        for (unsigned int i = 0; i < n_refinements; ++i)
          {
            execute_coarsening_and_refinement();
            initialize_solution();
          }
      }
    else
      {
        if (n_refinements)
          tria.refine_global(n_refinements);

        initialize();
        initialize_solution();
      }

    const auto output_result = [&](const VectorType  &vec,
                                   const double       t,
                                   const unsigned int step,
                                   std::string        label = "solution") {
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

      pcout << "outputing " << label << " at step = " << step << ", t = " << t
            << std::endl;

      std::string output = label + "." + std::to_string(step) + ".vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    };

    FEEvaluation<dim,
                 fe_degree,
                 n_points_1D,
                 n_comp,
                 Number,
                 VectorizedArrayType>
      eval(matrix_free);

    FEFaceEvaluation<dim,
                     fe_degree,
                     n_points_1D,
                     n_comp,
                     Number,
                     VectorizedArrayType>
      eval_m(matrix_free, true);

    FEFaceEvaluation<dim,
                     fe_degree,
                     n_points_1D,
                     n_comp,
                     Number,
                     VectorizedArrayType>
      eval_p(matrix_free, false);

    MassMatrix<dim, fe_degree, n_points_1D, n_comp, Number, VectorizedArrayType>
      mass_matrix(matrix_free);

    // Timestep estimates
    const Number dx2 = dx * dx;
    const Number dt_phi =
      2. / (1.5 * L(0, 1) *
            (0.64 * dx2 * k[0] * do_couple_phi_c + 4. / 3. * dx2 * B(0, 1) +
             2. * (5. + 1. / 3.) * A(0, 1)) /
            dx2);

    const Number dt_c = dx2 / (2 * 4 * D * (1 + std::abs(c_0[0] - c_0[1])));
    const Number dt =
      dtfac * (do_couple_phi_c ? std::min(dt_c, dt_phi) : dt_phi);

    // Params
    pcout << "dx     = " << dx << std::endl;
    pcout << "W      = " << W << std::endl;
    pcout << "D      = " << D << std::endl;
    pcout << "dt_phi = " << dt_phi << std::endl;
    pcout << "dt_c   = " << dt_c << std::endl;
    pcout << "dt     = " << dt << std::endl;
    pcout << std::endl;

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
    // throw std::runtime_error("STOP");

    // Preconditioner
    // PreconditionIdentity preconditioner;
    InverseDiagonalMatrix preconditioner(mass_matrix);

    // Count linear iterations
    unsigned int n_linear_iterations = 0;

    // RHS evaluation for the time-stepping
    const auto evaluate_rhs = [&](const double t, const VectorType &y) {
      (void)t;

      VectorType incr(n_comp);
      for (unsigned int c = 0; c < n_comp; ++c)
        matrix_free.initialize_dof_vector(incr.block(c));

      const VectorizedArrayType zeroes(0.0), ones(1.0);
      const VectorizedArrayType bulk_threshold(1e-10);
      const VectorizedArrayType zero_threshold(1e-18);
      const VectorizedArrayType phase_threshold = ones - bulk_threshold;

      auto cell_evaluator = [&](const auto &,
                                auto       &dst,
                                const auto &src,
                                auto        cells) {
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            eval.reinit(cell);
            eval.gather_evaluate(src,
                                 EvaluationFlags::values |
                                   EvaluationFlags::gradients);

            auto phi_avg = create_array<n_phi>(VectorizedArrayType(0.0));
            auto phi_grad_avg =
              create_array<n_phi>(Tensor<1, dim, VectorizedArrayType>());

            VectorizedArrayType                    is_bulk_cell(0.0);
            VectorizedArrayType                    nzs_cell(0.0);
            std::array<VectorizedArrayType, n_phi> ok_cell =
              create_array<n_phi>(VectorizedArrayType(0.0));

            for (unsigned int q = 0; q < eval.n_q_points; ++q)
              {
                const auto value    = eval.get_value(q);
                const auto gradient = eval.get_gradient(q);

                // Phase-field values and gradients
                for (unsigned int i = 0; i < n_phi; ++i)
                  {
                    phi_avg[i] += value[i];
                    phi_grad_avg[i] += gradient[i];

                    is_bulk_cell =
                      compare_and_apply_mask<SIMDComparison::less_than>(
                        std::abs(ones - value[i]),
                        bulk_threshold,
                        ones,
                        is_bulk_cell);

                    // Check value itself
                    ok_cell[i] =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        std::abs(value[i]), zero_threshold, ones, ok_cell[i]);

                    // Check gradients
                    ok_cell[i] =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        gradient[i].norm(), zero_threshold, ones, ok_cell[i]);
                  }
              }

            for (unsigned int i = 0; i < n_phi; ++i)
              {
                phi_avg[i] *= 1. / eval.n_q_points;
                phi_grad_avg[i] *= 1. / eval.n_q_points;

                nzs_cell += ok_cell[i];
              }

            for (unsigned int q = 0; q < eval.n_q_points; ++q)
              {
                const auto value    = eval.get_value(q);
                const auto gradient = eval.get_gradient(q);

                // Phase-field values and gradients
                std::array<VectorizedArrayType, n_phi>                 phi;
                std::array<Tensor<1, dim, VectorizedArrayType>, n_phi> phi_grad;
                for (unsigned int i = 0; i < n_phi; ++i)
                  {
                    phi[i]      = value[i];
                    phi_grad[i] = gradient[i];
                  }

                VectorizedArrayType c(0.0);
                if (do_couple_phi_c)
                  c = value[n_phi];

                // Precomputed divergences in FDM like manner (if needed)
                auto div_A_grad_phi =
                  create_array<n_phi>(VectorizedArrayType(0.0));
                (void)div_A_grad_phi; // disable warning

                VectorizedArrayType ca_cv(0.0), ca_cs(0.0), div_grad_c(0.0);
                (void)ca_cv;
                (void)ca_cs;
                (void)div_grad_c;

                // Kind of FDM
                if constexpr (is_dg && n_points_1D == 1)
                  {
                    for (unsigned int i = 0; i < n_phi; ++i)
                      phi_avg[i] = phi[i];

                    if (do_couple_phi_c)
                      {
                        ca_cv = calc_ca2<Number, k, c_0>(c, phi, 0);
                        ca_cs = calc_ca2<Number, k, c_0>(c, phi, 1);
                      }

                    Tensor<1, dim, unsigned int> num_faces;
                    unsigned int                 num_vals_to_avg = 1;
                    for (unsigned int face = 0;
                         face < GeometryInfo<dim>::faces_per_cell;
                         ++face)
                      {
                        const auto boundary_ids =
                          matrix_free.get_faces_by_cells_boundary_id(cell,
                                                                     face);

                        Assert(std::equal(
                                 boundary_ids.begin(),
                                 boundary_ids.begin() +
                                   matrix_free.n_active_entries_per_cell_batch(
                                     cell),
                                 boundary_ids.begin()),
                               ExcMessage("Boundary IDs of lanes differ."));

                        const auto boundary_id = boundary_ids[0];

                        if (boundary_id == numbers::internal_face_boundary_id)
                          {
                            eval_p.reinit(cell, face);
                            eval_p.gather_evaluate(src,
                                                   EvaluationFlags::values);

                            const auto value_p = eval_p.get_value(0);

                            const auto normal_vector = eval_p.normal_vector(q);

                            std::array<VectorizedArrayType, n_phi> nphi;

                            std::array<VectorizedArrayType, n_phi> phi_p;
                            for (unsigned int i = 0; i < n_phi; ++i)
                              phi_p[i] = value_p[i];

                            for (unsigned int i = 0; i < n_phi; ++i)
                              {
                                const auto numerical_flux =
                                  central_flux<dim>(phi[i],
                                                    phi_p[i],
                                                    normal_vector);
                                const auto numerical_grad = numerical_flux / dx;

                                phi_grad[i] += numerical_grad;

                                nphi[i] = 0.5 * (phi[i] + phi_p[i]);

                                phi_avg[i] += phi_p[i];
                              }

                            const auto Av = calcA(A, nphi);
                            for (unsigned int i = 0; i < n_phi; ++i)
                              div_A_grad_phi[i] += Av * (value_p[i] - phi[i]);

                            // Quantities for the c field
                            if (do_couple_phi_c)
                              {
                                VectorizedArrayType(0.0);
                                const auto c_p = value_p[n_phi];

                                const auto ca_cv_p =
                                  calc_ca2<Number, k, c_0>(c_p, phi_p, 0);
                                const auto ca_cs_p =
                                  calc_ca2<Number, k, c_0>(c_p, phi_p, 1);

                                const auto hphi0_p = hfunc2_0(phi_p);

                                div_grad_c +=
                                  D * (hphi0_p * (ca_cv_p - ca_cv) +
                                       (1. - hphi0_p) * (ca_cs_p - ca_cs));
                              }

                            num_faces[face / 2] += 1;

                            ++num_vals_to_avg;
                          }
                      }

                    // Second order derivative
                    for (unsigned int i = 0; i < n_phi; ++i)
                      div_A_grad_phi[i] *= 1. / (dx * dx);

                    if constexpr (do_couple_phi_c)
                      div_grad_c *= 1. / (dx * dx);

                    // Prevent division by zero
                    for (unsigned int d = 0; d < dim; ++d)
                      if (num_faces[d] == 0)
                        num_faces[d] = 1;

                    for (unsigned int i = 0; i < n_phi; ++i)
                      {
                        // average gradient
                        for (unsigned int d = 0; d < dim; ++d)
                          phi_grad[i][d] /= num_faces[d];

                        // average phi values
                        phi_avg[i] /= num_vals_to_avg;
                      }
                  }

                VectorizedArrayType                    is_bulk(0.0);
                VectorizedArrayType                    nzs(0.0);
                std::array<VectorizedArrayType, n_phi> ok;
                for (unsigned int i = 0; i < n_phi; ++i)
                  {
                    is_bulk = compare_and_apply_mask<SIMDComparison::less_than>(
                      std::abs(ones - phi[i]), bulk_threshold, ones, is_bulk);

                    // Check value itself
                    ok[i] =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        std::abs(phi[i]), zero_threshold, ones, zeroes);

                    // Check gradients
                    ok[i] =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        phi_grad[i].norm(), zero_threshold, ones, ok[i]);

                    nzs += ok[i];
                  }
                // DEBUG
                // is_bulk = ones;


                const auto &nzs_eval     = nzs_cell;
                const auto &ok_eval      = ok_cell;
                const auto &is_bulk_eval = is_bulk_cell;

                const auto   invnzs = 1. / nzs_eval;
                const Number ifac   = 1. / (n_phi - 1.);

                // Precomputed partial derivatives
                std::array<std::pair<VectorizedArrayType,
                                     Tensor<1, dim, VectorizedArrayType>>,
                           n_phi>
                  dFdphi_arr;
                for (unsigned int i = 0; i < n_phi; ++i)
                  dFdphi_arr[i] = dFdphi<Number, k, c_0, do_couple_phi_c>(
                    A, B, phi, phi_grad, c, i);

                Tensor<1, n_comp, VectorizedArrayType> value_result;
                Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                  gradient_result;

                auto prefacs = create_array<n_phi>(
                  create_array<n_phi>(VectorizedArrayType(0.0)));

                auto prefacs_grad = create_array<n_phi>(
                  create_array<n_phi>(Tensor<1, dim, VectorizedArrayType>()));

                // unsigned int upper_tri_offset = 0;

                // TEST - use averaged phi values
                // auto phi_eval      = phi;
                // auto phi_grad_eval = phi_grad;
                auto &phi_eval      = phi_avg;
                auto &phi_grad_eval = phi_grad_avg;

                for (unsigned int i = 0; i < n_phi; ++i)
                  {
                    VectorizedArrayType pre_i =
                      std::abs(phi_eval[i] / (1. - phi_eval[i]));

                    pre_i =
                      compare_and_apply_mask<SIMDComparison::greater_than>(
                        phi_eval[i], phase_threshold, ones, pre_i);

                    for (unsigned int j = i + 1; j < n_phi; ++j)
                      {
                        VectorizedArrayType pre_j =
                          std::abs(phi_eval[j] / (1. - phi_eval[j]));

                        pre_j =
                          compare_and_apply_mask<SIMDComparison::greater_than>(
                            phi_eval[j], phase_threshold, ones, pre_j);

                        const auto val = pre_i * pre_j;

                        prefacs[i][i] += val;
                        prefacs[j][j] += val;

                        prefacs[i][j] = val;
                        prefacs[j][i] = val;

                        const auto prod = phi_eval[i] / (1. - phi_eval[i]) *
                                          phi_eval[j] / (1. - phi_eval[j]);

                        VectorizedArrayType sgn(1.0);
                        sgn =
                          compare_and_apply_mask<SIMDComparison::greater_than>(
                            prod, zeroes, sgn, -sgn);

                        // Gradient terms
                        Tensor<1, dim, VectorizedArrayType> pre_grad_ij0 =
                          phi_eval[j] * phi_grad_eval[i] /
                            (std::pow(ones - phi_eval[i], 2.) *
                             (ones - phi_eval[j])) +
                          phi_eval[i] * phi_grad_eval[j] /
                            (std::pow(ones - phi_eval[j], 2.) *
                             (ones - phi_eval[i]));
                        (void)pre_grad_ij0;

                        Tensor<1, dim, VectorizedArrayType> pre_grad_ij =
                          sgn *
                          (phi_eval[j] * (ones - phi_eval[j]) *
                             phi_grad_eval[i] +
                           phi_eval[i] * (ones - phi_eval[i]) *
                             phi_grad_eval[j]) /
                          (std::pow(ones - phi_eval[i], 2.) *
                           std::pow(ones - phi_eval[j], 2.));

                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            pre_grad_ij[d] = compare_and_apply_mask<
                              SIMDComparison::greater_than>(phi_eval[i],
                                                            phase_threshold,
                                                            zeroes,
                                                            pre_grad_ij[d]);

                            pre_grad_ij[d] = compare_and_apply_mask<
                              SIMDComparison::greater_than>(phi_eval[j],
                                                            phase_threshold,
                                                            zeroes,
                                                            pre_grad_ij[d]);
                          }

                        prefacs_grad[i][i] += pre_grad_ij;
                        prefacs_grad[j][j] += pre_grad_ij;

                        prefacs_grad[i][j] = pre_grad_ij;
                        prefacs_grad[j][i] = pre_grad_ij;
                      }
                  }
                /*
                                  const auto prefac_outer =
                                    compare_and_apply_mask<SIMDComparison::greater_than>(
                                      is_bulk, zeroes, invnzs, ones);
                */

                // upper_tri_offset = 0;
                for (unsigned int i = 0; i < n_phi; ++i)
                  {
                    const auto prefac_outer =
                      (ones - is_bulk_eval) +
                      is_bulk_eval * invnzs * ok_eval[i];

                    for (unsigned int j = 0; j < n_phi; ++j)
                      if (i != j)
                        {
                          const auto prefac_inner_left =
                            (ones - is_bulk_eval) * ifac * prefacs[i][i] +
                            is_bulk_eval * ok_eval[i] * ok_eval[j];

                          const auto prefac_inner_right =
                            (ones - is_bulk) * prefacs[i][j] +
                            is_bulk * ok_eval[i] * ok_eval[j];

                          value_result[i] +=
                            -L(i, j) *
                            (prefac_inner_left * dFdphi_arr[i].first -
                             prefac_inner_right * dFdphi_arr[j].first);

                          if constexpr (is_dg && n_points_1D == 1)
                            {
                              // Sign differs from the true FEM case
                              value_result[i] +=
                                L(i, j) *
                                (prefac_inner_left * div_A_grad_phi[i] -
                                 prefac_inner_right * div_A_grad_phi[j]);
                            }
                          else
                            {
                              // Additional term since P_ab is nonlinear
                              const auto prefac_inner_left_grad =
                                (ones - is_bulk_eval) * ifac *
                                prefacs_grad[i][i];

                              const auto prefac_inner_right_grad =
                                (ones - is_bulk_eval) * prefacs_grad[i][j];

                              value_result[i] +=
                                -L(i, j) *
                                (prefac_inner_left_grad * dFdphi_arr[i].second -
                                 prefac_inner_right_grad *
                                   dFdphi_arr[j].second);

                              gradient_result[i] +=
                                -L(i, j) *
                                (prefac_inner_left * dFdphi_arr[i].second -
                                 prefac_inner_right * dFdphi_arr[j].second);
                            }
                        }

                    // DEBUG individual terms
                    // value_result[i] = div_A_grad_phi[i];
                    // value_result[i]    = dFdphi_arr[i].first;
                    // gradient_result[i] = dFdphi_arr[i].second;

                    value_result[i] *= prefac_outer;
                    gradient_result[i] *= prefac_outer;

                    // upper_tri_offset += n_phi - i - 1;
                  }

                if constexpr (do_couple_phi_c)
                  {
                    if constexpr (is_dg && n_points_1D == 1)
                      {
                        value_result[n_phi] = div_grad_c;
                      }
                    else
                      {
                        const auto &c_grad = gradient[n_phi];

                        // Evaluate hphi0
                        const auto hphi0 = hfunc2_0(phi);

                        // Some auxiliary variables
                        const auto grad_ca_0 = calc_grad_ca2<Number, k, c_0>(
                          c, c_grad, phi, phi_grad, 0);
                        const auto grad_ca_1 = calc_grad_ca2<Number, k, c_0>(
                          c, c_grad, phi, phi_grad, 1);

                        gradient_result[n_phi] =
                          -D * (hphi0 * grad_ca_0 + (1. - hphi0) * grad_ca_1);
                      }
                  }

                eval.submit_value(value_result, q);
                eval.submit_gradient(gradient_result, q);
              }
            eval.integrate_scatter(EvaluationFlags::values |
                                     EvaluationFlags::gradients,
                                   dst);
          }
      };

      if constexpr (is_dg && n_points_1D == 1)
        {
          // TODO: Interface evaluator for DG
          auto face_evaluator = [&](const auto &,
                                    auto       &dst,
                                    const auto &src,
                                    auto        face_range) {
            for (unsigned int face = face_range.first; face < face_range.second;
                 ++face)
              {
                eval_p.reinit(face);
                eval_p.gather_evaluate(src, EvaluationFlags::values);

                eval_m.reinit(face);
                eval_m.gather_evaluate(src, EvaluationFlags::values);

                for (const unsigned int q : eval_m.quadrature_point_indices())
                  {
                    const auto value_p = eval_p.get_value(q);
                    const auto value_m = eval_m.get_value(q);

                    const auto normal_vector = eval_m.normal_vector(q);

                    std::array<VectorizedArrayType, n_phi> phi_p;
                    std::array<VectorizedArrayType, n_phi> phi_m;
                    std::array<Tensor<1, dim, VectorizedArrayType>, n_phi>
                      phi_grad;
                    for (unsigned int i = 0; i < n_phi; ++i)
                      {
                        phi_p[i] = value_p[i];
                        phi_m[i] = value_m[i];

                        const auto numerical_flux =
                          central_flux<dim>(phi_p[i], phi_m[i], normal_vector);
                        const auto numerical_grad = numerical_flux / dx;

                        std::cout << "face = " << face << " -> ";
                        std::cout << "q = " << q << " -> ";
                        // std::cout << " phi_p[" << i << "] = " << phi_p[i];
                        // std::cout << " phi_m[" << i << "] = " << phi_m[i];
                        std::cout << " numerical_grad[" << i
                                  << "] = " << numerical_grad << std::endl;

                        phi_grad[i] = numerical_grad;
                      }

                    // eval_m.submit_value(-numerical_flux, q);
                    // eval_p.submit_value(numerical_flux, q);
                  }
              }
          };
          (void)face_evaluator;

          matrix_free.template loop_cell_centric<VectorType, VectorType>(
            cell_evaluator, rhs, y, true);
        }
      else
        {
          matrix_free.template cell_loop<VectorType, VectorType>(cell_evaluator,
                                                                 rhs,
                                                                 y,
                                                                 true);
        }

      // rhs.block(0).print(std::cout);
      // output_result(rhs, 1.0, 1);
      // throw std::runtime_error("STOP");

      static unsigned int local_step_counter = 0;

      if constexpr (print_debug)
        output_result(rhs, t, local_step_counter, "rhs");

      {
        ReductionControl     reduction_control;
        SolverCG<VectorType> solver(reduction_control);
        solver.solve(mass_matrix, incr, rhs, preconditioner);

        for (unsigned int c = 0; c < n_comp; ++c)
          constraint.distribute(incr.block(c));

        if constexpr (print)
          pcout << "t = " << t << ": "
                << "incr.l2_norm() = " << incr.l2_norm() << " | "
                << "n_linear_iter = " << reduction_control.last_step()
                << std::endl;

        n_linear_iterations += reduction_control.last_step();
      }

      if constexpr (print_debug)
        {
          output_result(incr, t, local_step_counter, "incr");
        }
      local_step_counter++;

      return incr;
    };

    // const auto test_rhs = evaluate_rhs(0, solution);
    // output_result(test_rhs, 1.0, 1);
    // throw std::runtime_error("STOP");

    // time loop
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

        // Temporary sanitizing - DISABLE
        if constexpr (false)
          {
            for (unsigned int b = 0; b < solution.n_blocks(); ++b)
              for (auto &v : solution.block(b))
                v = std::min(std::max(v, 0.), 1.);

            for (unsigned int i = 0; i < solution.block(0).size(); ++i)
              {
                auto phi_sum = 0;
                for (unsigned int b = 0; b < solution.n_blocks(); ++b)
                  phi_sum += solution.block(b)[i];

                if (phi_sum == 0.)
                  phi_sum = 1.;
                for (unsigned int b = 0; b < solution.n_blocks(); ++b)
                  solution.block(b)[i] *= 1. / phi_sum;
              }
          }

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

    pcout << "Execution time: " << elapsed.count() << "s" << std::endl;
    pcout << "n_linear_iterations: " << n_linear_iterations << std::endl;
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

  /* Use DG0
    This is an experimental approach, since it is a very much simplified DG
    implementation that mainly mimics FDM, i.e. it is supposed to work with DG0
    only. Furthermore, not all terms are yet properly evaluated for this case.
    Though the gradients are somewhat evaluated for phases only, the
    corresponding divergence terms are not yet evaluated correctly.
  */
  // Test<2, 0, true> runner;

  // Use CG
  Test<2, 1, false> runner;

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
  char     *n_refs_char = get_cmd_option(argv, argv + argc, "-r");
  const int n_refs =
    n_refs_char ? std::stoi(n_refs_char) : 0; // default to 0 refinements

  // Number of steps
  char              *n_steps_char = get_cmd_option(argv, argv + argc, "-s");
  const unsigned int n_steps =
    n_steps_char ? std::stoi(n_steps_char) : 0; // default to 0 steps

  // Number of steps to generate output
  char              *n_output_char = get_cmd_option(argv, argv + argc, "-o");
  const unsigned int n_steps_output =
    n_output_char ? std::stoi(n_output_char) : 100; // default to 100 steps

  // Whether to perform AMR
  const bool do_amr =
    cmd_option_exists(argv, argv + argc, "-amr"); // default to false

  pcout << std::endl;
  pcout << "Number of additional refinements: " << n_refs << std::endl;
  pcout << "Number of steps:                  " << n_steps << std::endl;
  pcout << "Output at every:                  " << n_steps_output << std::endl;
  pcout << std::endl;

  runner.run(method, n_refs, n_steps, n_steps_output, do_amr);

  return 0;
}