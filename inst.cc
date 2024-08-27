#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <deal.II/trilinos/nox.h>
#include <deal.II/trilinos/nox.templates.h>
#include <pf-applications/grain_tracker/representation.h>

using namespace dealii;

template class TrilinosWrappers::NOXSolver<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template void
internal::AffineConstraintsImplementation::set_zero_all(
  const std::vector<types::global_dof_index> &,
  LinearAlgebra::distributed::DynamicBlockVector<double> &);

// Explicitly export instantiations to make polymorphic serialization work
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<2>)
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<3>)
