#define MAX_SINTERING_GRAINS 2
#define FE_DEGREE 1
#define N_Q_POINTS_1D FE_DEGREE + 1

#include <deal.II/base/mpi.h>

#include <pf-applications/sintering/operator_sintering_generic.h>

#include <pf-applications/tests/residual_tester.h>

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

  Test::calc_residual<dim, Number, VectorizedArrayType, NonLinearOperator>(
    true);
}
