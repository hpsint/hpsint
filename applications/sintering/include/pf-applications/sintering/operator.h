#pragma once

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <pf-applications/matrix_free/tools.h>

#include <fstream>

template <typename T>
using n_grains_t = decltype(std::declval<T const>().n_grains());

template <typename T>
using n_grains_to_n_components_t =
  decltype(std::declval<T const>().n_grains_to_n_components(
    std::declval<const unsigned int>()));

template <typename T>
constexpr bool has_n_grains_method =
  dealii::internal::is_supported_operation<n_grains_t, T>
    &&dealii::internal::is_supported_operation<n_grains_to_n_components_t, T>;

// clang-format off
/**
 * Macro that converts a runtime number (n_components() or n_grains())
 * to constant expressions that can be used for templating and calles
 * the provided function with the two parameters: 1) number of
 * components and 2) number of grains (if it makes sence; else -1).
 *
 * The relation between number of components and number of grains
 * is encrypted in the method T::n_grains_to_n_components().
 * 
 * The function can be used the following way:
 * ```
 * #define OPERATION(c, d) std::cout << c << " " << d << std::endl;
 * EXPAND_OPERATIONS(OPERATION);
 * #undef OPERATION
 * ```
 */
#define EXPAND_OPERATIONS(OPERATION)                                                                                  \
  if constexpr(has_n_grains_method<T>)                                                                                \
    {                                                                                                                 \
      constexpr int max_grains = MAX_SINTERING_GRAINS;                                                                \
      const unsigned int n_grains = static_cast<const T&>(*this).n_grains();                                          \
      AssertIndexRange(n_grains, max_grains + 1);                                                                     \
      switch (n_grains)                                                                                               \
        {                                                                                                             \
          case  2: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  2)), std::min(max_grains,  2)); break; \
          case  3: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  3)), std::min(max_grains,  3)); break; \
          case  4: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  4)), std::min(max_grains,  4)); break; \
          case  5: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  5)), std::min(max_grains,  5)); break; \
          case  6: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  6)), std::min(max_grains,  6)); break; \
          case  7: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  7)), std::min(max_grains,  7)); break; \
          case  8: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  8)), std::min(max_grains,  8)); break; \
          case  9: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  9)), std::min(max_grains,  9)); break; \
          case 10: OPERATION(T::n_grains_to_n_components(std::min(max_grains, 10)), std::min(max_grains, 10)); break; \
          default:                                                                                                    \
            AssertThrow(false, ExcNotImplemented());                                                                  \
        }                                                                                                             \
    }                                                                                                                 \
  else                                                                                                                \
    {                                                                                                                 \
      constexpr int max_components = MAX_SINTERING_GRAINS +2 ;                                                        \
      AssertIndexRange(this->n_components(), max_components + 1);                                                     \
      switch (this->n_components())                                                                                   \
        {                                                                                                             \
          case  1: OPERATION(std::min(max_components,  1), -1); break;                                                \
          case  2: OPERATION(std::min(max_components,  2), -1); break;                                                \
          case  3: OPERATION(std::min(max_components,  3), -1); break;                                                \
          case  4: OPERATION(std::min(max_components,  4), -1); break;                                                \
          case  5: OPERATION(std::min(max_components,  5), -1); break;                                                \
          case  6: OPERATION(std::min(max_components,  6), -1); break;                                                \
          case  7: OPERATION(std::min(max_components,  7), -1); break;                                                \
          case  8: OPERATION(std::min(max_components,  8), -1); break;                                                \
          case  9: OPERATION(std::min(max_components,  9), -1); break;                                                \
          case 10: OPERATION(std::min(max_components, 10), -1); break;                                                \
          default:                                                                                                    \
            AssertThrow(false, ExcNotImplemented());                                                                  \
        }                                                                                                             \
  }
// clang-format on

namespace Sintering
{
  using namespace dealii;
  template <int dim, typename VectorizedArrayType>
  class MobilityScalar
  {
  protected:
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;

  public:
    MobilityScalar(const double Mvol,
                   const double Mvap,
                   const double Msurf,
                   const double Mgb)
      : Mvol(Mvol)
      , Mvap(Mvap)
      , Msurf(Msurf)
      , Mgb(Mgb)
    {}

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    M(const VectorizedArrayType &                c,
      const VectorTypeValue &                    etas,
      const Tensor<1, dim, VectorizedArrayType> &c_grad,
      const VectorTypeGradient &                 etas_grad) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType etaijSum = 0.0;
      for (const auto &etai : etas)
        {
          for (const auto &etaj : etas)
            {
              if (etai != etaj)
                {
                  etaijSum += (*etai) * (*etaj);
                }
            }
        }

      VectorizedArrayType phi =
        cl * cl * cl * (10.0 - 15.0 * cl + 6.0 * cl * cl);
      std::for_each(phi.begin(), phi.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType M = Mvol * phi + Mvap * (1.0 - phi) +
                              Msurf * cl * (1.0 - cl) + Mgb * etaijSum;

      return M;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_dc(const VectorizedArrayType &                c,
          const VectorTypeValue &                    etas,
          const Tensor<1, dim, VectorizedArrayType> &c_grad,
          const VectorTypeGradient &                 etas_grad) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      (void)etas;
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType dphidc = 30.0 * cl * cl * (1.0 - 2.0 * cl + cl * cl);
      VectorizedArrayType dMdc =
        Mvol * dphidc - Mvap * dphidc + Msurf * (1.0 - 2.0 * cl);

      return dMdc;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      (void)c;
      (void)c_grad;
      (void)mu_grad;

      return Tensor<2, dim, VectorizedArrayType>();
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    dM_detai(const VectorizedArrayType &                c,
             const VectorTypeValue &                    etas,
             const Tensor<1, dim, VectorizedArrayType> &c_grad,
             const VectorTypeGradient &                 etas_grad,
             unsigned int                               index_i) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      (void)c;
      (void)c_grad;
      (void)etas_grad;

      VectorizedArrayType etajSum = 0;
      for (unsigned int j = 0; j < etas.size(); j++)
        {
          if (j != index_i)
            {
              etajSum += *etas[j];
            }
        }

      auto MetajSum = 2.0 * Mgb * etajSum;

      return MetajSum;
    }
  };



  template <int dim, typename VectorizedArrayType>
  class MobilityTensorial
  {
  protected:
    double Mvol;
    double Mvap;
    double Msurf;
    double Mgb;

  public:
    MobilityTensorial(const double Mvol,
                      const double Mvap,
                      const double Msurf,
                      const double Mgb)
      : Mvol(Mvol)
      , Mvap(Mvap)
      , Msurf(Msurf)
      , Mgb(Mgb)
    {}

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          M(const VectorizedArrayType &                c,
                            const VectorTypeValue &                    etas,
                            const Tensor<1, dim, VectorizedArrayType> &c_grad,
                            const VectorTypeGradient &                 etas_grad) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType phi =
        cl * cl * cl * (10.0 - 15.0 * cl + 6.0 * cl * cl);
      std::for_each(phi.begin(), phi.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> M =
        unitMatrix(Mvol * phi + Mvap * (1.0 - phi));

      // Surface anisotropic part
      VectorizedArrayType fsurf = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      M += projectorMatrix(nc, fsurf);

      // GB diffusion part
      for (unsigned int i = 0; i < etas.size(); i++)
        {
          for (unsigned int j = 0; j < etas.size(); j++)
            {
              if (i != j)
                {
                  VectorizedArrayType fgb = Mgb * (*etas[i]) * (*etas[j]);
                  Tensor<1, dim, VectorizedArrayType> etaGradDiff =
                    (*etas_grad[i]) - (*etas_grad[j]);
                  Tensor<1, dim, VectorizedArrayType> neta =
                    unitVector(etaGradDiff);
                  M += projectorMatrix(neta, fgb);
                }
            }
        }

      return M;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dc(const VectorizedArrayType &                c,
                                const VectorTypeValue &                    etas,
                                const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                const VectorTypeGradient &                 etas_grad) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      (void)etas;
      (void)etas_grad;

      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType dphidc = 30.0 * cl * cl * (1.0 - 2.0 * cl + cl * cl);

      // Volumetric and vaporization parts, the same as for isotropic
      Tensor<2, dim, VectorizedArrayType> dMdc =
        unitMatrix((Mvol - Mvap) * dphidc);

      // Surface part
      VectorizedArrayType fsurf  = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      VectorizedArrayType dfsurf = Msurf * 2. * cl * (1. - cl) * (1. - 2. * cl);
      for (unsigned int i = 0; i < fsurf.size(); i++)
        {
          if (fsurf[i] < 1e-6)
            {
              dfsurf[i] = 0.;
            }
        }
      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      dMdc += projectorMatrix(nc, dfsurf);

      return dMdc;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_dgrad_c(const VectorizedArrayType &                c,
                                     const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                     const Tensor<1, dim, VectorizedArrayType> &mu_grad) const
    {
      VectorizedArrayType cl = c;
      std::for_each(cl.begin(), cl.end(), [](auto &val) {
        val = val > 1.0 ? 1.0 : (val < 0.0 ? 0.0 : val);
      });

      VectorizedArrayType fsurf = Msurf * (cl * cl) * ((1. - cl) * (1. - cl));
      VectorizedArrayType nrm   = c_grad.norm();

      for (unsigned int i = 0; i < nrm.size(); i++)
        {
          if (nrm[i] < 1e-4 || fsurf[i] < 1e-6)
            {
              fsurf[i] = 0.;
            }
          if (nrm[i] < 1e-10)
            {
              nrm[i] = 1.;
            }
        }

      Tensor<1, dim, VectorizedArrayType> nc = unitVector(c_grad);
      Tensor<2, dim, VectorizedArrayType> M  = projectorMatrix(nc, 1. / nrm);

      Tensor<2, dim, VectorizedArrayType> T =
        unitMatrix(mu_grad * nc) + outer_product(nc, mu_grad);
      T *= -fsurf;

      return T * M;
    }

    template <typename VectorTypeValue, typename VectorTypeGradient>
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          dM_detai(const VectorizedArrayType &                c,
                                   const VectorTypeValue &                    etas,
                                   const Tensor<1, dim, VectorizedArrayType> &c_grad,
                                   const VectorTypeGradient &                 etas_grad,
                                   unsigned int                               index_i) const
    {
      static_assert(std::is_same<typename VectorTypeValue::value_type,
                                 const VectorizedArrayType *>::value);
      static_assert(
        std::is_same<typename VectorTypeGradient::value_type,
                     const Tensor<1, dim, VectorizedArrayType> *>::value);

      (void)c;
      (void)c_grad;

      dealii::Tensor<2, dim, VectorizedArrayType> M;

      for (unsigned int j = 0; j < etas.size(); j++)
        {
          if (j != index_i)
            {
              VectorizedArrayType                 fgb = 2. * Mgb * (*etas[j]);
              Tensor<1, dim, VectorizedArrayType> etaGradDiff =
                (*etas_grad[index_i]) - (*etas_grad[j]);
              Tensor<1, dim, VectorizedArrayType> neta =
                unitVector(etaGradDiff);
              M += projectorMatrix(neta, fgb);
            }
        }

      return M;
    }

  private:
    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
                          unitMatrix(const VectorizedArrayType &fac = 1.) const
    {
      Tensor<2, dim, VectorizedArrayType> I;

      for (unsigned int d = 0; d < dim; d++)
        {
          I[d][d] = fac;
        }

      return I;
    }

    DEAL_II_ALWAYS_INLINE Tensor<1, dim, VectorizedArrayType>
    unitVector(const Tensor<1, dim, VectorizedArrayType> &vec) const
    {
      VectorizedArrayType nrm = vec.norm();
      VectorizedArrayType filter;

      Tensor<1, dim, VectorizedArrayType> n = vec;

      for (unsigned int i = 0; i < nrm.size(); i++)
        {
          if (nrm[i] > 1e-4)
            {
              filter[i] = 1.;
            }
          else
            {
              nrm[i] = 1.;
            }
        }

      n /= nrm;
      n *= filter;

      return n;
    }

    DEAL_II_ALWAYS_INLINE Tensor<2, dim, VectorizedArrayType>
    projectorMatrix(const Tensor<1, dim, VectorizedArrayType> vec,
                    const VectorizedArrayType &               fac = 1.) const
    {
      auto tensor = unitMatrix() - dealii::outer_product(vec, vec);
      tensor *= fac;

      return tensor;
    }
  };

  template <unsigned int n, std::size_t p>
  class PowerHelper
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T *, n> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + std::pow(*b, static_cast<double>(p));
        });
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::vector<T *> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + std::pow(*b, static_cast<double>(p));
        });
    }
  };

  template <>
  class PowerHelper<2, 2>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T *, 2> &etas)
    {
      return (*etas[0]) * (*etas[0]) + (*etas[1]) * (*etas[1]);
    }
  };

  template <>
  class PowerHelper<2, 3>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T *, 2> &etas)
    {
      return (*etas[0]) * (*etas[0]) * (*etas[0]) +
             (*etas[1]) * (*etas[1]) * (*etas[1]);
    }
  };

  template <class T>
  class SizeHelper;

  template <class T, std::size_t n>
  class SizeHelper<std::array<T, n>>
  {
  public:
    static const std::size_t size = n;
  };

  template <class T>
  class SizeHelper<std::vector<T>>
  {
  public:
    static const std::size_t size = 0;
  };


  template <typename VectorizedArrayType>
  class FreeEnergy
  {
  private:
    double A;
    double B;

  public:
    FreeEnergy(double A, double B)
      : A(A)
      , B(B)
    {}

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c, const VectorType &etas) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
             B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                  (-4.0 * c + 8.0) * etaPower3Sum +
                  3.0 * (etaPower2Sum * etaPower2Sum));
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c, const VectorType &etas) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return A * (c * c) * (2.0 * c - 2.0) +
             2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
             B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             unsigned int               index_i) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      const auto &etai = *etas[index_i];

      return B * (3.0 * (etai * etai) * (4.0 * c - 8.0) +
                  2.0 * etai * (-6.0 * c + 6.0) + 12.0 * etai * (etaPower2Sum));
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dc2(const VectorizedArrayType &c, const VectorType &etas) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      (void)etas;

      return 2.0 * A * (c * c) + 4.0 * A * c * (2.0 * c - 2.0) +
             2.0 * A * ((-c + 1.0) * (-c + 1.0)) + 2.0 * B;
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dcdetai(const VectorizedArrayType &c,
                const VectorType &         etas,
                unsigned int               index_i) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      (void)c;

      const auto &etai = *etas[index_i];

      return B * (12.0 * (etai * etai) - 12.0 * etai);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               unsigned int               index_i) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      const auto &etai = *etas[index_i];

      return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                  24.0 * (etai * etai) + 12.0 * etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detaidetaj(const VectorizedArrayType &c,
                   const VectorType &         etas,
                   unsigned int               index_i,
                   unsigned int               index_j) const
    {
      static_assert(std::is_same<typename VectorType::value_type,
                                 const VectorizedArrayType *>::value);

      (void)c;

      const auto &etai = *etas[index_i];
      const auto &etaj = *etas[index_j];

      return 24.0 * B * etai * etaj;
    }
  };
} // namespace Sintering

namespace Sintering
{
  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class OperatorBase : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    static const int dimension = dim;

    OperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number> &                   constraints,
      const unsigned int                                  dof_index,
      const std::string                                   label = "")
      : matrix_free(matrix_free)
      , constraints(constraints)
      , dof_index(dof_index)
      , label(label)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(label != "")
      , do_timing(true)
    {}

    virtual ~OperatorBase() = default;

    virtual void
    clear()
    {
      this->system_matrix.clear();
      this->block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    virtual unsigned int
    n_components() const = 0;

    virtual unsigned int
    n_unique_components() const
    {
      return n_components();
    }

    const DoFHandler<dim> &
    get_dof_handler() const
    {
      return matrix_free.get_dof_handler(dof_index);
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst, dof_index);
    }

    void
    initialize_dof_vector(BlockVectorType &dst) const
    {
      dst.reinit(this->n_components());
      for (unsigned int c = 0; c < this->n_components(); ++c)
        matrix_free.initialize_dof_vector(dst.block(c), dof_index);
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler(dof_index).n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    bool
    set_timing(const bool do_timing) const
    {
      const bool old  = this->do_timing;
      this->do_timing = do_timing;
      return old;
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult", this->do_timing);

      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
          system_matrix.vmult(dst, src);
        }
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult", this->do_timing);

      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
#define OPERATION(c, d)                                                     \
  MyMatrixFreeTools::cell_loop_wrapper(this->matrix_free,                   \
                                       &OperatorBase::do_vmult_range<c, d>, \
                                       this,                                \
                                       dst,                                 \
                                       src,                                 \
                                       true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
          if (src_.size() == 0 || dst_.size() == 0)
            {
              const auto partitioner = get_system_partitioner();

              src_.reinit(partitioner);
              dst_.reinit(partitioner);
            }

          VectorTools::merge_components_fast(src, src_); // TODO
          this->vmult(dst_, src_);
          VectorTools::split_up_components_fast(dst_, dst); // TODO
        }
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      AssertThrow(false, ExcNotImplemented());

      this->vmult(dst, src);
    }

    void
    compute_inverse_diagonal(VectorType &diagonal) const
    {
      MyScope scope(this->timer, label + "::diagonal", this->do_timing);

      matrix_free.initialize_dof_vector(diagonal, dof_index);

#define OPERATION(c, d)                                                 \
  MatrixFreeTools::compute_diagonal(matrix_free,                        \
                                    diagonal,                           \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      for (auto &i : diagonal)
        i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    void
    compute_inverse_diagonal(BlockVectorType &diagonal) const
    {
      MyScope scope(this->timer, label + "::diagonal", this->do_timing);

      AssertDimension(
        matrix_free.get_dof_handler(dof_index).get_fe().n_components(), 1);

      diagonal.reinit(this->n_components());
      for (unsigned int b = 0; b < this->n_components(); ++b)
        matrix_free.initialize_dof_vector(diagonal.block(b), dof_index);

#define OPERATION(c, d)                                                 \
  MatrixFreeTools::compute_diagonal(matrix_free,                        \
                                    diagonal,                           \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      for (unsigned int b = 0; b < this->n_components(); ++b)
        for (auto &i : diagonal.block(b))
          i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }

    std::shared_ptr<Utilities::MPI::Partitioner>
    get_system_partitioner() const
    {
      const auto partitioner_scalar =
        this->matrix_free.get_vector_partitioner(dof_index);

      IndexSet is(this->n_components());
      is.add_range(0, this->n_components());

      return std::make_shared<Utilities::MPI::Partitioner>(
        partitioner_scalar->locally_owned_range().tensor_product(is),
        partitioner_scalar->ghost_indices().tensor_product(is),
        partitioner_scalar->get_mpi_communicator());
    }

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer, label + "::matrix::sp", this->do_timing);

          system_matrix.clear();

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          DoFHandler<dim> dof_handler(
            this->matrix_free.get_dof_handler(dof_index).get_triangulation());
          dof_handler.distribute_dofs(
            FESystem<dim>(this->matrix_free.get_dof_handler(dof_index).get_fe(),
                          this->n_components()));

          constraints_for_matrix.clear();
          constraints_for_matrix.reinit(
            DoFTools::extract_locally_relevant_dofs(dof_handler));
          DoFTools::make_hanging_node_constraints(dof_handler,
                                                  constraints_for_matrix);
          constraints_for_matrix.close();

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler,
                                          dsp,
                                          constraints_for_matrix);
          dsp.compress();

          system_matrix.reinit(dsp);

          this->pcout << std::endl;
          this->pcout << "Create sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - NNZ: " << system_matrix.n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      label + "::matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          {
            system_matrix = 0.0; // clear existing content
          }

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    constraints_for_matrix,             \
                                    system_matrix,                      \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      return system_matrix;
    }

    void
    clear_system_matrix() const
    {
      system_matrix.clear();
      block_system_matrix.clear();
      src_.reinit(0);
      dst_.reinit(0);
    }

    const std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>> &
    get_block_system_matrix() const
    {
      const bool system_matrix_is_empty = block_system_matrix.size() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer,
                        this->label + "::block_matrix::sp",
                        this->do_timing);

          AssertDimension(this->matrix_free.get_dof_handler(dof_index)
                            .get_fe()
                            .n_components(),
                          1);

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);
          dsp.compress();

          block_system_matrix.resize(this->n_components());
          for (unsigned int b = 0; b < this->n_components(); ++b)
            {
              block_system_matrix[b] =
                std::make_shared<TrilinosWrappers::SparseMatrix>();
              block_system_matrix[b]->reinit(dsp);
            }

          this->pcout << std::endl;
          this->pcout << "Create block sparsity pattern (" << this->label
                      << ") with:" << std::endl;
          this->pcout << " - number of blocks: " << this->n_components()
                      << std::endl;
          this->pcout << " - NNZ:              "
                      << block_system_matrix[0]->n_nonzero_elements()
                      << std::endl;
          this->pcout << std::endl;
        }

      {
        MyScope scope(this->timer,
                      label + "::block_matrix::compute",
                      this->do_timing);

        if (system_matrix_is_empty == false)
          for (unsigned int b = 0; b < this->n_components(); ++b)
            *block_system_matrix[b] = 0.0; // clear existing content

#define OPERATION(c, d)                                                 \
  MyMatrixFreeTools::compute_matrix(matrix_free,                        \
                                    this->constraints,                  \
                                    block_system_matrix,                \
                                    &OperatorBase::do_vmult_cell<c, d>, \
                                    this,                               \
                                    dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      return block_system_matrix;
    }

  protected:
    template <int n_comp, int n_grains>
    void
    do_vmult_cell(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                   EvaluationFlags::EvaluationFlags::gradients);

      static_cast<const T &>(*this).template do_vmult_kernel<n_comp, n_grains>(
        phi);

      phi.integrate(EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_range(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, dof_index);

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          static_cast<const T &>(*this)
            .template do_vmult_kernel<n_comp, n_grains>(phi);

          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

  protected:
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;
    const AffineConstraints<Number> &                   constraints;
    mutable AffineConstraints<Number>                   constraints_for_matrix;

    const unsigned int dof_index;

    const std::string label;

    mutable TrilinosWrappers::SparseMatrix system_matrix;
    mutable std::vector<std::shared_ptr<TrilinosWrappers::SparseMatrix>>
      block_system_matrix;

    ConditionalOStream    pcout;
    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    mutable VectorType src_, dst_;
  };



  template <int dim, typename VectorizedArrayType>
  struct SinteringOperatorData
  {
    using Number = typename VectorizedArrayType::value_type;

    SinteringOperatorData(const Number A,
                          const Number B,
                          const Number Mvol,
                          const Number Mvap,
                          const Number Msurf,
                          const Number Mgb,
                          const Number L,
                          const Number kappa_c,
                          const Number kappa_p)
      : free_energy(A, B)
      , mobility(Mvol, Mvap, Msurf, Mgb)
      , L(L)
      , kappa_c(kappa_c)
      , kappa_p(kappa_p)
    {}

    const FreeEnergy<VectorizedArrayType> free_energy;

    // Choose MobilityScalar or MobilityTensorial here:
    const MobilityScalar<dim, VectorizedArrayType> mobility;
    // const MobilityTensorial<dim, VectorizedArrayType> mobility;

    const Number L;
    const Number kappa_c;
    const Number kappa_p;

    Number dt;

    Table<3, VectorizedArrayType> &
    get_nonlinear_values()
    {
      return nonlinear_values;
    }

    Table<3, VectorizedArrayType> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients()
    {
      return nonlinear_gradients;
    }

    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
    }

    void
    set_n_components(const unsigned int number_of_components)
    {
      this->number_of_components = number_of_components;
    }

    unsigned int
    n_components() const
    {
      return number_of_components;
    }

    unsigned int
    n_grains() const
    {
      return number_of_components - 2;
    }

    void
    fill_quadrature_point_values(
      const MatrixFree<dim, Number, VectorizedArrayType> &          matrix_free,
      const LinearAlgebra::distributed::DynamicBlockVector<Number> &src)
    {
      AssertDimension(src.n_blocks(), this->n_components());

      const unsigned n_cells             = matrix_free.n_cell_batches();
      const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

      nonlinear_values.reinit(
        {n_cells, n_quadrature_points, this->n_components()});
      nonlinear_gradients.reinit(
        {n_cells, n_quadrature_points, this->n_components()});

      FECellIntegrator<dim, 1, Number, VectorizedArrayType> phi(matrix_free);

      src.update_ghost_values();

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          phi.reinit(cell);

          for (unsigned int c = 0; c < this->n_components(); ++c)
            {
              phi.read_dof_values_plain(src.block(c));
              phi.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  nonlinear_values(cell, q, c)    = phi.get_value(q);
                  nonlinear_gradients(cell, q, c) = phi.get_gradient(q);
                }
            }
        }

      src.zero_out_ghost_values();
    }

  private:
    mutable Table<3, VectorizedArrayType> nonlinear_values;
    mutable Table<3, dealii::Tensor<1, dim, VectorizedArrayType>>
      nonlinear_gradients;

    unsigned int number_of_components;
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          SinteringOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = SinteringOperator<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const AffineConstraints<Number> &                      constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const bool                                             matrix_based)
      : OperatorBase<dim,
                     Number,
                     VectorizedArrayType,
                     SinteringOperator<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          0,
          "sintering_op")
      , data(data)
      , matrix_based(matrix_based)
    {}

    ~SinteringOperator()
    {}

    void
    evaluate_nonlinear_residual(BlockVectorType &      dst,
                                const BlockVectorType &src) const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

#define OPERATION(c, d)                                       \
  this->matrix_free.cell_loop(                                \
    &SinteringOperator::do_evaluate_nonlinear_residual<c, d>, \
    this,                                                     \
    dst,                                                      \
    src,                                                      \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    void
    set_previous_solution(const BlockVectorType &src) const
    {
      Assert(src.has_ghost_elements() == false, ExcInternalError());

      AssertThrow(src.is_globally_compatible(
                    this->matrix_free.get_vector_partitioner()),
                  ExcInternalError());

      this->old_solution = src;
      this->old_solution.update_ghost_values();

      AssertThrow(this->old_solution.is_globally_compatible(
                    this->matrix_free.get_vector_partitioner()),
                  ExcInternalError());
    }

    const BlockVectorType &
    get_previous_solution() const
    {
      AssertThrow(this->old_solution.is_globally_compatible(
                    this->matrix_free.get_vector_partitioner()),
                  ExcInternalError());

      this->old_solution.zero_out_ghost_values();
      return this->old_solution;
    }

    void
    do_update()
    {
      if (matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &
    get_data() const
    {
      return data;
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors(DataOut<dim> &               data_out,
                        const BlockVectorType &      vec,
                        const std::set<std::string> &fields_list) const
    {
      AssertDimension(n_comp - 2, n_grains);

      // Possible output options
      enum OutputFields
      {
        FieldBnds,
        FieldDt,
        FieldD2f,
        FieldM,
        FieldDM,
        FieldKappa,
        FieldL
      };

      const std::array<std::tuple<std::string, OutputFields, unsigned int>, 7>
        possible_entries = {
          {{"bnds", FieldBnds, 1},
           {"dt", FieldDt, 1},
           {"d2f", FieldD2f, 1 + 2 * n_grains + n_grains * (n_grains - 1) / 2},
           {"M", FieldM, 1},
           {"dM", FieldDM, 2 + n_grains},
           {"kappa", FieldKappa, 2},
           {"L", FieldL, 1}}};

      // A better design is possible, but at the moment this is sufficient
      std::array<bool, 7> entries_mask;
      entries_mask.fill(false);

      unsigned int n_entries = 0;

      for (unsigned int i = 0; i < possible_entries.size(); i++)
        {
          const auto &entry = possible_entries[i];
          if (fields_list.count(std::get<0>(entry)))
            {
              entries_mask[std::get<1>(entry)] = true;
              n_entries += std::get<2>(entry);
            }
        }

      if (n_entries == 0)
        return;

      std::vector<VectorType> data_vectors(n_entries);

      for (auto &data_vector : data_vectors)
        this->matrix_free.initialize_dof_vector(data_vector, this->dof_index);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> fe_eval_all(
        this->matrix_free, this->dof_index);
      FECellIntegrator<dim, 1, Number, VectorizedArrayType> fe_eval(
        this->matrix_free, this->dof_index);

      MatrixFreeOperators::
        CellwiseInverseMassMatrix<dim, -1, 1, Number, VectorizedArrayType>
          inverse_mass_matrix(fe_eval);

      AlignedVector<VectorizedArrayType> buffer(fe_eval.n_q_points * n_entries);

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;

      vec.update_ghost_values();

      std::vector<VectorizedArrayType> temp(n_entries);

      for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches();
           ++cell)
        {
          fe_eval_all.reinit(cell);
          fe_eval.reinit(cell);

          fe_eval_all.reinit(cell);
          fe_eval_all.read_dof_values_plain(vec);
          fe_eval_all.evaluate(EvaluationFlags::values |
                               EvaluationFlags::gradients);

          for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
            {
              const auto val  = fe_eval_all.get_value(q);
              const auto grad = fe_eval_all.get_gradient(q);

              const auto &c       = val[0];
              const auto &c_grad  = grad[0];
              const auto &mu_grad = grad[1];

              std::array<const VectorizedArrayType *, n_grains> etas;
              std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
                etas_grad;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]      = &val[2 + ig];
                  etas_grad[ig] = &grad[2 + ig];
                }

              unsigned int counter = 0;

              if (entries_mask[FieldBnds])
                {
                  temp[counter++] = PowerHelper<n_grains, 2>::power_sum(etas);
                }

              if (entries_mask[FieldDt])
                {
                  temp[counter++] = VectorizedArrayType(data.dt);
                }

              if (entries_mask[FieldD2f])
                {
                  temp[counter++] = free_energy.d2f_dc2(c, etas);

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy.d2f_dcdetai(c, etas, ig);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] = free_energy.d2f_detai2(c, etas, ig);
                    }

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                        {
                          temp[counter++] =
                            free_energy.d2f_detaidetaj(c, etas, ig, jg);
                        }
                    }
                }

              if (entries_mask[FieldM])
                {
                  temp[counter++] = mobility.M(c, etas, c_grad, etas_grad);
                }

              if (entries_mask[FieldDM])
                {
                  temp[counter++] =
                    (mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad)
                      .norm();
                  temp[counter++] =
                    (mobility.dM_dgrad_c(c, c_grad, mu_grad)).norm();

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      temp[counter++] =
                        (mobility.dM_detai(c, etas, c_grad, etas_grad, ig) *
                         mu_grad)
                          .norm();
                    }
                }

              if (entries_mask[FieldKappa])
                {
                  temp[counter++] = VectorizedArrayType(kappa_c);
                  temp[counter++] = VectorizedArrayType(kappa_p);
                }

              if (entries_mask[FieldL])
                {
                  temp[counter++] = VectorizedArrayType(L);
                }

              for (unsigned int c = 0; c < n_entries; ++c)
                buffer[c * fe_eval.n_q_points + q] = temp[c];
            }

          for (unsigned int c = 0; c < n_entries; ++c)
            {
              inverse_mass_matrix.transform_from_q_points_to_basis(
                1,
                buffer.data() + c * fe_eval.n_q_points,
                fe_eval.begin_dof_values());

              fe_eval.set_dof_values_plain(data_vectors[c]);
            }
        }

      // TODO: remove once FEEvaluation::set_dof_values_plain()
      // sets the values of constrainging DoFs in the case of PBC
      for (unsigned int c = 0; c < n_entries; ++c)
        this->constraints.distribute(data_vectors[c]);

      vec.zero_out_ghost_values();

      // Write names of fields
      std::vector<std::string> names;
      if (entries_mask[FieldBnds])
        {
          names.push_back("bnds");
        }

      if (entries_mask[FieldDt])
        {
          names.push_back("dt");
        }

      if (entries_mask[FieldD2f])
        {
          names.push_back("d2f_dc2");

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("d2f_dcdeta" + std::to_string(ig));
            }

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("d2f_deta" + std::to_string(ig) + "2");
            }

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                {
                  names.push_back("d2f_deta" + std::to_string(ig) + "deta" +
                                  std::to_string(jg));
                }
            }
        }

      if (entries_mask[FieldM])
        {
          names.push_back("M");
        }

      if (entries_mask[FieldDM])
        {
          names.push_back("nrm_dM_dc_x_mu_grad");
          names.push_back("nrm_dM_dgrad_c");

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              names.push_back("nrm_dM_deta" + std::to_string(ig) +
                              "_x_mu_grad");
            }
        }

      if (entries_mask[FieldKappa])
        {
          names.push_back("kappa_c");
          names.push_back("kappa_p");
        }

      if (entries_mask[FieldL])
        {
          names.push_back("L");
        }

      // Add data to output
      for (unsigned int c = 0; c < n_entries; ++c)
        {
          data_out.add_data_vector(this->matrix_free.get_dof_handler(
                                     this->dof_index),
                                   data_vectors[c],
                                   names[c]);
        }
    }

    void
    add_data_vectors(DataOut<dim> &               data_out,
                     const BlockVectorType &      vec,
                     const std::set<std::string> &fields_list) const
    {
#define OPERATION(c, d) \
  this->do_add_data_vectors<c, d>(data_out, vec, fields_list);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    unsigned int
    n_components() const override
    {
      return data.n_components();
    }

    unsigned int
    n_grains() const
    {
      return this->n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2;
    }

    template <int n_comp, int n_grains>
    void
    do_vmult_kernel(
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> &phi) const
    {
      AssertDimension(n_comp - 2, n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / this->data.dt;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto &val  = nonlinear_values[cell][q];
          const auto &grad = nonlinear_gradients[cell][q];

          const auto &c       = val[0];
          const auto &c_grad  = grad[0];
          const auto &mu_grad = grad[1];

          std::array<const VectorizedArrayType *, n_grains> etas;
          std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
            etas_grad;

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              etas[ig]      = &val[2 + ig];
              etas_grad[ig] = &grad[2 + ig];
            }

          Tensor<1, n_comp, VectorizedArrayType> value_result;
          Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
            gradient_result;

          value_result[0] = phi.get_value(q)[0] * dt_inv;
          value_result[1] = -phi.get_value(q)[1] +
                            free_energy.d2f_dc2(c, etas) * phi.get_value(q)[0];

          gradient_result[0] =
            mobility.M(c, etas, c_grad, etas_grad) * phi.get_gradient(q)[1] +
            mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad *
              phi.get_value(q)[0] +
            mobility.dM_dgrad_c(c, c_grad, mu_grad) * phi.get_gradient(q)[0];

          gradient_result[1] = kappa_c * phi.get_gradient(q)[0];

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[1] +=
                free_energy.d2f_dcdetai(c, etas, ig) * phi.get_value(q)[ig + 2];

              value_result[ig + 2] =
                phi.get_value(q)[ig + 2] * dt_inv +
                L * free_energy.d2f_dcdetai(c, etas, ig) * phi.get_value(q)[0] +
                L * free_energy.d2f_detai2(c, etas, ig) *
                  phi.get_value(q)[ig + 2];

              gradient_result[0] +=
                mobility.dM_detai(c, etas, c_grad, etas_grad, ig) * mu_grad *
                phi.get_value(q)[ig + 2];

              gradient_result[ig + 2] =
                L * kappa_p * phi.get_gradient(q)[ig + 2];

              for (unsigned int jg = 0; jg < n_grains; ++jg)
                {
                  if (ig != jg)
                    {
                      value_result[ig + 2] +=
                        L * free_energy.d2f_detaidetaj(c, etas, ig, jg) *
                        phi.get_value(q)[jg + 2];
                    }
                }
            }

          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      AssertDimension(n_comp - 2, n_grains);

      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi_old(
        matrix_free, this->dof_index);
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / this->data.dt;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_old.reinit(cell);
          phi.reinit(cell);

          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          // get values from old solution
          phi_old.read_dof_values_plain(old_solution);
          phi_old.evaluate(EvaluationFlags::EvaluationFlags::values);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              const auto val     = phi.get_value(q);
              const auto val_old = phi_old.get_value(q);
              const auto grad    = phi.get_gradient(q);

              auto &c      = val[0];
              auto &mu     = val[1];
              auto &c_old  = val_old[0];
              auto &c_grad = grad[0];

              std::array<const VectorizedArrayType *, n_grains> etas;
              std::array<const Tensor<1, dim, VectorizedArrayType> *, n_grains>
                etas_grad;

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  etas[ig]      = &val[2 + ig];
                  etas_grad[ig] = &grad[2 + ig];
                }

              Tensor<1, n_comp, VectorizedArrayType> value_result;
              Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                gradient_result;

              // CH equations
              value_result[0] = (c - c_old) * dt_inv;
              value_result[1] = -mu + free_energy.df_dc(c, etas);
              gradient_result[0] =
                mobility.M(c, etas, c_grad, etas_grad) * grad[1];
              gradient_result[1] = kappa_c * grad[0];

              // AC equations
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[2 + ig] =
                    (val[2 + ig] - val_old[2 + ig]) * dt_inv +
                    L * free_energy.df_detai(c, etas, ig);

                  gradient_result[2 + ig] = L * kappa_p * grad[2 + ig];
                }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &data;

    mutable BlockVectorType old_solution;

    const bool matrix_based;
  };

} // namespace Sintering
