#define MAX_SINTERING_GRAINS 2
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

#include <deal.II/base/mpi.h>

#include <pf-applications/sintering/operator_sintering_generic.h>

#include "include/tangent_tester.h"

using namespace dealii;
using namespace Sintering;

using Number              = double;
using VectorizedArrayType = VectorizedArray<Number>;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  constexpr int dim = 2;

  using NonLinearOperator =
    SinteringOperatorGeneric<dim, Number, VectorizedArrayType>;

  const bool   enable_rbm = true;
  const double tol_abs    = 1e2;
  const double tol_rel    = 1e-3;

  Test::check_tangent<dim, Number, VectorizedArrayType, NonLinearOperator>(
    enable_rbm, tol_abs, tol_rel);
}