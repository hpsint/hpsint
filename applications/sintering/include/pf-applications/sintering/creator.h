
#include <pf-applications/sintering/operator_sintering_generic.h>

#include <pf-applications/structural/material.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename BlockVectorType,
            typename NonLinearOperator>
  inline auto
  create_sintering_operator(
    const MatrixFree<dim, Number, VectorizedArrayType> &     matrix_free,
    const AffineConstraints<Number> &                        constraints,
    const SinteringOperatorData<dim, VectorizedArrayType> &  sintering_data,
    const TimeIntegration::SolutionHistory<BlockVectorType> &solution_history,
    const AdvectionMechanism<dim, Number, VectorizedArrayType>
      &          advection_mechanism,
    const bool   matrix_based,
    const bool   use_tensorial_mobility_gradient_on_the_fly = false,
    const double E                                          = 1.0,
    const double nu                                         = 0.25,
    const Structural::MaterialPlaneType type =
      Structural::MaterialPlaneType::none,
    std::function<Tensor<1, dim, VectorizedArrayType>(
      const Point<dim, VectorizedArrayType>)> loading = {})
  {
    (void)advection_mechanism;
    (void)use_tensorial_mobility_gradient_on_the_fly;
    (void)E;
    (void)nu;
    (void)loading;
    (void)type;

    if constexpr (std::is_same_v<
                    NonLinearOperator,
                    SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>)

      return SinteringOperatorGeneric<dim, Number, VectorizedArrayType>(
        matrix_free,
        constraints,
        sintering_data,
        solution_history,
        advection_mechanism,
        matrix_based,
        use_tensorial_mobility_gradient_on_the_fly);
  }

} // namespace Sintering