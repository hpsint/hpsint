#pragma once

#include <deal.II/base/utilities.h>
namespace Sintering
{
  using namespace dealii;

  template <unsigned int n, std::size_t p>
  class PowerHelper
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      T initial = 0.0;

      for (unsigned int i = 0; i < n; ++i)
        initial += Utilities::fixed_power<p>(etas[i]);

      return initial;
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, n> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::vector<T> &etas)
    {
      T initial = 0.0;

      return std::accumulate(
        etas.begin(), etas.end(), initial, [](auto a, auto b) {
          return std::move(a) + Utilities::fixed_power<p>(b);
        });
    }
  };

  template <>
  class PowerHelper<2, 2>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] + etas[1] * etas[1];
    }
  };

  template <>
  class PowerHelper<2, 3>
  {
  public:
    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const T *etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
    }

    template <typename T>
    DEAL_II_ALWAYS_INLINE static T
    power_sum(const std::array<T, 2> &etas)
    {
      return etas[0] * etas[0] * etas[0] + etas[1] * etas[1] * etas[1];
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
    f(const VectorizedArrayType &c,
      const VectorType &         etas,
      const VectorizedArrayType &etaPower2Sum,
      const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * ((-c + 1.0) * (-c + 1.0)) +
             B * ((c * c) + (-6.0 * c + 6.0) * etaPower2Sum -
                  (-4.0 * c + 8.0) * etaPower3Sum +
                  3.0 * (etaPower2Sum * etaPower2Sum));
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    f(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return f(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c,
          const VectorType &         etas,
          const VectorizedArrayType &etaPower2Sum,
          const VectorizedArrayType &etaPower3Sum) const
    {
      (void)etas;

      return A * (c * c) * (2.0 * c - 2.0) +
             2.0 * A * c * ((-c + 1.0) * (-c + 1.0)) +
             B * (2.0 * c - 6.0 * etaPower2Sum + 4.0 * etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_dc(const VectorizedArrayType &c, const VectorType &etas) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);
      const auto etaPower3Sum = PowerHelper<n, 3>::power_sum(etas);

      return df_dc(c, etas, etaPower2Sum, etaPower3Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             const VectorizedArrayType &etaPower2Sum,
             unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return etai * (B * 12.0) *
             (etai * (1.0 * c - 2.0) + (-c + 1.0) + etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    df_detai(const VectorizedArrayType &c,
             const VectorType &         etas,
             unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return df_detai(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_dc2(const VectorizedArrayType &c, const VectorType &etas) const
    {
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
      (void)c;

      const auto &etai = etas[index_i];

      return (B * 12.0) * etai * (etai - 1.0);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               const VectorizedArrayType &etaPower2Sum,
               unsigned int               index_i) const
    {
      const auto &etai = etas[index_i];

      return B * (12.0 - 12.0 * c + 2.0 * etai * (12.0 * c - 24.0) +
                  24.0 * (etai * etai) + 12.0 * etaPower2Sum);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detai2(const VectorizedArrayType &c,
               const VectorType &         etas,
               unsigned int               index_i) const
    {
      const std::size_t n = SizeHelper<VectorType>::size;

      const auto etaPower2Sum = PowerHelper<n, 2>::power_sum(etas);

      return d2f_detai2(c, etas, etaPower2Sum, index_i);
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE VectorizedArrayType
    d2f_detaidetaj(const VectorizedArrayType &c,
                   const VectorType &         etas,
                   unsigned int               index_i,
                   unsigned int               index_j) const
    {
      (void)c;

      const auto &etai = etas[index_i];
      const auto &etaj = etas[index_j];

      return 24.0 * B * etai * etaj;
    }

    template <typename VectorType>
    DEAL_II_ALWAYS_INLINE void
    apply_d2f_detaidetaj(const VectorType &         L,
                         const VectorizedArrayType *etas,
                         const unsigned int         n_grains,
                         const VectorizedArrayType *value,
                         VectorizedArrayType *      value_result) const
    {
      if (n_grains <= 1)
        return; // nothing to do

      VectorizedArrayType temp = etas[0] * value[0];

      for (unsigned int ig = 1; ig < n_grains; ++ig)
        temp += etas[ig] * value[ig];

      for (unsigned int ig = 0; ig < n_grains; ++ig)
        value_result[ig] +=
          (L * 24.0 * B) * etas[ig] * (temp - etas[ig] * value[ig]);
    }
  };

} // namespace Sintering
