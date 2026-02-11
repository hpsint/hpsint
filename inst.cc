#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/affine_constraints.templates.h>

#ifdef DEAL_II_WITH_SUNDIALS
#  include <deal.II/sundials/arkode.h>
#  include <deal.II/sundials/arkode.templates.h>
#  include <deal.II/sundials/n_vector.h>
#  include <deal.II/sundials/n_vector.templates.h>
#  include <deal.II/sundials/sunlinsol_wrapper.h>
#  include <deal.II/sundials/sunlinsol_wrapper.templates.h>
#endif

#include <deal.II/trilinos/nox.h>
#include <deal.II/trilinos/nox.templates.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/grain_tracker/representation.h>

using namespace dealii;

template class TrilinosWrappers::NOXSolver<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template void
internal::AffineConstraintsImplementation::set_zero_all(
  const std::vector<types::global_dof_index> &,
  LinearAlgebra::distributed::DynamicBlockVector<double> &);

// SUNDIALS wrapper instantiations
#ifdef DEAL_II_WITH_SUNDIALS
template LinearAlgebra::distributed::DynamicBlockVector<double> *
  SUNDIALS::internal::unwrap_nvector<
    LinearAlgebra::distributed::DynamicBlockVector<double>>(N_Vector);
template const LinearAlgebra::distributed::DynamicBlockVector<double> *
  SUNDIALS::internal::unwrap_nvector_const<
    LinearAlgebra::distributed::DynamicBlockVector<double>>(N_Vector);

template class SUNDIALS::internal::NVectorView<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;
template class SUNDIALS::internal::NVectorView<
  const LinearAlgebra::distributed::DynamicBlockVector<double>>;

template struct SUNDIALS::SundialsOperator<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template struct SUNDIALS::SundialsPreconditioner<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template class SUNDIALS::internal::LinearSolverWrapper<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;

template class SUNDIALS::ARKode<
  LinearAlgebra::distributed::DynamicBlockVector<double>>;
#endif

// Explicitly export instantiations to make polymorphic serialization work
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<2>)
BOOST_CLASS_EXPORT(GrainTracker::RepresentationSpherical<3>)
BOOST_CLASS_EXPORT(GrainTracker::RepresentationElliptical<2>)
BOOST_CLASS_EXPORT(GrainTracker::RepresentationElliptical<3>)
