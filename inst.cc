#include <pf-applications/lac/dynamic_block_vector.h>

#include <deal.II/trilinos/nox.h>
#include <deal.II/trilinos/nox.templates.h>

using namespace dealii;

template class TrilinosWrappers::NOXSolver<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;