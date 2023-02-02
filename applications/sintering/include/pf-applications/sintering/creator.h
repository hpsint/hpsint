
#include <pf-applications/sintering/operator_sintering_coupled_diffusion.h>
#include <pf-applications/sintering/operator_sintering_coupled_wang.h>
#include <pf-applications/sintering/operator_sintering_generic.h>

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
    const double E                                          = 1.0,
    const double nu                                         = 0.25,
    const bool   use_tensorial_mobility_gradient_on_the_fly = false,
    std::function<Tensor<1, dim, VectorizedArrayType>(
      const Point<dim, VectorizedArrayType>)> loading       = {})
  {
    (void)advection_mechanism;
    (void)E;
    (void)nu;
    (void)loading;

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

    else if constexpr (
      std::is_same_v<
        NonLinearOperator,
        SinteringOperatorCoupledWang<dim, Number, VectorizedArrayType>>)

      return SinteringOperatorCoupledWang<dim, Number, VectorizedArrayType>(
        matrix_free,
        constraints,
        sintering_data,
        solution_history,
        advection_mechanism,
        matrix_based,
        E,
        nu,
        loading);

    else if constexpr (
      std::is_same_v<
        NonLinearOperator,
        SinteringOperatorCoupledDiffusion<dim, Number, VectorizedArrayType>>)

      return SinteringOperatorCoupledDiffusion<dim,
                                               Number,
                                               VectorizedArrayType>(
        matrix_free,
        constraints,
        sintering_data,
        solution_history,
        matrix_based,
        E,
        nu);
  }

} // namespace Sintering