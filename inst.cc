#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <deal.II/trilinos/nox.h>
#include <deal.II/trilinos/nox.templates.h>

using namespace dealii;

template class TrilinosWrappers::NOXSolver<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template void
internal::AffineConstraintsImplementation::set_zero_all(
  const std::vector<types::global_dof_index> &,
  LinearAlgebra::distributed::DynamicBlockVector<double> &);
