#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "operators.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

static unsigned int likwid_counter = 0;


using namespace dealii;

struct Parameters
{
  unsigned int dim                  = 2;
  unsigned int n_global_refinements = 1;

  unsigned int fe_degree           = 2;
  unsigned int n_quadrature_points = 0;
  unsigned int n_subdivisions      = 1;
  std::string  fe_type             = "FE_Q";

  unsigned int level = 2;

  unsigned int n_repetitions = 10;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("dim", dim);
    prm.add_parameter("n global refinements", n_global_refinements);

    prm.add_parameter("fe type",
                      fe_type,
                      "",
                      Patterns::Selection("FE_Q|FE_Q_iso_Q1"));
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("n quadrature points", n_quadrature_points);
    prm.add_parameter("n subdivisions", n_subdivisions);

    prm.add_parameter("level", level);

    prm.add_parameter("n repetitions", n_repetitions);
  }
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const unsigned int component)
    : component(component)
  {}


  virtual double
  value(const Point<dim> &p, const unsigned int = 0) const override
  {
    if (component == 0)
      return p[0];
    else
      return 0.0;
  }

private:
  const unsigned int component;
};

template <int n_components,
          int dim,
          typename Number,
          typename VectorizedArrayType>
std::shared_ptr<ProjectionOperatorBase<Number>>
create_op(const unsigned int                                  level,
          const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
{
  const auto &si = matrix_free.get_shape_info().data.front();

  const unsigned int fe_degree     = si.fe_degree;
  const unsigned int n_q_points_1d = si.n_q_points_1d;

  if ((fe_degree == 1) && (n_q_points_1d == 2))
    return std::make_shared<
      ProjectionOperator<dim, 1, 2, n_components, Number, VectorizedArrayType>>(
      matrix_free, level);
  if ((fe_degree == 2) && (n_q_points_1d == 4))
    return std::make_shared<
      ProjectionOperator<dim, 2, 4, n_components, Number, VectorizedArrayType>>(
      matrix_free, level);
  if ((fe_degree == 3) && (n_q_points_1d == 6))
    return std::make_shared<
      ProjectionOperator<dim, 3, 6, n_components, Number, VectorizedArrayType>>(
      matrix_free, level);

  AssertThrow(false, ExcNotImplemented());

  return std::make_shared<
    ProjectionOperator<dim, -1, 0, n_components, Number, VectorizedArrayType>>(
    matrix_free, level);
}

template <int dim, typename Number, typename VectorizedArrayType>
std::shared_ptr<ProjectionOperatorBase<Number>>
create_op(const unsigned int                                  n_components,
          const unsigned int                                  level,
          const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free)
{
  if (n_components == 1)
    return create_op<1>(level, matrix_free);
  else if (n_components == 2)
    return create_op<2>(level, matrix_free);
  else if (n_components == 3)
    return create_op<3>(level, matrix_free);
  else if (n_components == 4)
    return create_op<4>(level, matrix_free);
  else if (n_components == 5)
    return create_op<5>(level, matrix_free);
  else if (n_components == 6)
    return create_op<6>(level, matrix_free);
  else if (n_components == 7)
    return create_op<7>(level, matrix_free);
  else if (n_components == 8)
    return create_op<8>(level, matrix_free);
  else if (n_components == 9)
    return create_op<9>(level, matrix_free);
  else if (n_components == 10)
    return create_op<10>(level, matrix_free);
  else if (n_components == 11)
    return create_op<11>(level, matrix_free);
  else if (n_components == 12)
    return create_op<12>(level, matrix_free);
#if 0
  else if (n_components == 13)
    return create_op<13>(level, matrix_free);
  else if (n_components == 14)
    return create_op<14>(level, matrix_free);
  else if (n_components == 15)
    return create_op<15>(level, matrix_free);
  else if (n_components == 16)
    return create_op<16>(level, matrix_free);
#endif

  AssertThrow(false, ExcNotImplemented());

  return create_op<1>(level, matrix_free);
}

template <int dim, typename Number, typename VectorizedArrayType>
void
test(const Parameters &params, ConvergenceTable &table)
{
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  const auto fe_type              = params.fe_type;
  const auto fe_degree            = params.fe_degree;
  const auto n_subdivisions       = params.n_subdivisions;
  const auto n_global_refinements = params.n_global_refinements;
  const auto print_l2_norm        = false;

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>>    quadrature;
  MappingQ1<dim>                      mapping_q1;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  if (fe_type == "FE_Q")
    {
      AssertThrow(n_subdivisions == 1, ExcInternalError());

      fe = std::make_unique<FE_Q<dim>>(fe_degree);

      const unsigned int n_quadrature_points = params.n_quadrature_points > 0 ?
                                                 params.n_quadrature_points :
                                                 (fe_degree + 1);

      quadrature = std::make_unique<QGauss<dim>>(n_quadrature_points);
    }
  else if (fe_type == "FE_Q_iso_Q1")
    {
      AssertThrow(fe_degree == 1, ExcInternalError());

      fe = std::make_unique<FE_Q_iso_Q1<dim>>(n_subdivisions);
      quadrature =
        std::make_unique<QIterated<dim>>(QGauss<1>(2), n_subdivisions);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }


  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  AffineConstraints<Number> constraints;

  MappingQCache<dim> mapping(1);


  mapping.initialize(
    mapping_q1,
    tria,
    [](const auto &, const auto &point) {
      Point<dim> result;

      if (true) // TODO
        return result;

      for (unsigned int d = 0; d < dim; ++d)
        result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                    std::sin(numbers::PI * point[d]) * 0.01;

      return result;
    },
    true);

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.overlap_communication_computation = false;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, *quadrature, additional_data);

  const auto run = [&](const auto fu) {
    // warm up
    for (unsigned int i = 0; i < 10; ++i)
      fu();

#ifdef LIKWID_PERFMON
    const auto add_padding = [](const int value) -> std::string {
      if (value < 10)
        return "000" + std::to_string(value);
      if (value < 100)
        return "00" + std::to_string(value);
      if (value < 1000)
        return "0" + std::to_string(value);
      if (value < 10000)
        return "" + std::to_string(value);

      AssertThrow(false, ExcInternalError());

      return "";
    };

    const std::string likwid_label =
      "likwid_" + add_padding(likwid_counter); // TODO
    likwid_counter++;
#endif

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(likwid_label.c_str());
#endif

    const auto timer = std::chrono::system_clock::now();

    for (unsigned int i = 0; i < params.n_repetitions; ++i)
      fu();

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(likwid_label.c_str());
#endif

    const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now() - timer)
                          .count() /
                        1e9;

    return time;
  };

  for (unsigned int n_components = 1; n_components <= 12; ++n_components)
    {
      table.add_value("dim", dim);
      table.add_value("fe_type", fe_type);
      table.add_value("fe_degree", fe_degree);
      table.add_value("n_quadrature_points",
                      quadrature->get_tensor_basis()[0].size());
      table.add_value("n_subdivisions", n_subdivisions);
      table.add_value("n_global_refinements", n_global_refinements);
      table.add_value("n_repetitions", params.n_repetitions);
      table.add_value("n_dofs", dof_handler.n_dofs());
      table.add_value("n_components", n_components);

      // version 2: vectorial (block system)
      const auto projection_operator =
        create_op(n_components, params.level, matrix_free);

      BlockVectorType src, dst;
      projection_operator->initialize_dof_vector(src);
      projection_operator->initialize_dof_vector(dst);

      for (unsigned int i = 0; i < n_components; ++i)
        VectorTools::interpolate(dof_handler,
                                 RightHandSide<dim>(i),
                                 src.block(i));

      unsigned int counter = 0;

      const auto time = run([&]() {
        projection_operator->vmult(dst, src);

        if (print_l2_norm && (counter++ == 0))
          pcout << dst.l2_norm() << std::endl;
      });

      table.add_value("t_vector", time / n_components);
      table.set_scientific("t_vector", true);
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  AssertThrow(argc >= 2, ExcInternalError());

  ConvergenceTable table;

  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;

  for (int i = 1; i < argc; ++i)
    {
      Parameters params;
      params.parse(std::string(argv[i]));

      if (params.dim == 2)
        test<2, Number, VectorizedArrayType>(params, table);
      else if (params.dim == 3)
        test<3, Number, VectorizedArrayType>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
