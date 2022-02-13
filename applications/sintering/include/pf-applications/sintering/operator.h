#pragma once

#include <fstream>

template <typename T>
using n_grains_t = decltype(std::declval<T const>().n_grains());

template <typename T>
constexpr bool has_n_grains =
  dealii::internal::is_supported_operation<n_grains_t, T>;

// clang-format off
#define EXPAND_OPERATIONS(OPERATION)                                                                                  \
  if constexpr(has_n_grains<T>)                                                                                       \
    {                                                                                                                 \
      constexpr int max_grains = SINTERING_GRAINS;                                                                    \
      const unsigned int n_grains = static_cast<const T&>(*this).n_grains();                                          \
      AssertIndexRange(n_grains, max_grains + 1);                                                                     \
      switch (n_grains)                                                                                               \
        {                                                                                                             \
          case  1: OPERATION(T::n_grains_to_n_components(std::min(max_grains,  1)), std::min(max_grains,  1)); break; \
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
            Assert(false, ExcNotImplemented());                                                                       \
        }                                                                                                             \
    }                                                                                                                 \
  else                                                                                                                \
    {                                                                                                                 \
      constexpr int max_components = SINTERING_GRAINS +2 ;                                                            \
      AssertIndexRange(this->n_components(), max_components + 1);                                                     \
      switch (this->n_components())                                                                                   \
        {                                                                                                             \
          case  1: OPERATION(std::min(max_components,  1), 0); break;                                                 \
          case  2: OPERATION(std::min(max_components,  2), 0); break;                                                 \
          case  3: OPERATION(std::min(max_components,  3), 0); break;                                                 \
          case  4: OPERATION(std::min(max_components,  4), 0); break;                                                 \
          case  5: OPERATION(std::min(max_components,  5), 0); break;                                                 \
          case  6: OPERATION(std::min(max_components,  6), 0); break;                                                 \
          case  7: OPERATION(std::min(max_components,  7), 0); break;                                                 \
          case  8: OPERATION(std::min(max_components,  8), 0); break;                                                 \
          case  9: OPERATION(std::min(max_components,  9), 0); break;                                                 \
          case 10: OPERATION(std::min(max_components, 10), 0); break;                                                 \
          default:                                                                                                    \
            Assert(false, ExcNotImplemented());                                                                       \
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



  template <int dim, typename Number, typename VectorizedArrayType, typename T>
  class OperatorBase : public Subscriptor
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    static const int dimension = dim;

    OperatorBase(
      const MatrixFree<dim, Number, VectorizedArrayType> &  matrix_free,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const unsigned int                                    dof_index,
      const std::string                                     label = "")
      : matrix_free(matrix_free)
      , constraints(*constraints[dof_index])
      , dof_index(dof_index)
      , components_number(
          matrix_free.get_dof_handler(dof_index).get_fe().n_components())
      , label(label)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    {}

    virtual ~OperatorBase()
    {
      if (timer.get_summary_data(TimerOutput::OutputData::total_wall_time)
            .size() > 0)
        timer.print_wall_time_statistics(MPI_COMM_WORLD);
    }

    virtual void
    clear()
    {
      this->system_matrix.clear();
    }

    unsigned int
    n_components() const
    {
      return components_number;
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

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, label + "::vmult");


      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
#define OPERATION(c, d)        \
  this->matrix_free.cell_loop( \
    &OperatorBase::do_vmult_range<c, d>, this, dst, src, true);
          EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
        }
      else
        {
          system_matrix.vmult(dst, src);
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
      MyScope scope(this->timer, label + "::diagonal");

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

    const TrilinosWrappers::SparseMatrix &
    get_system_matrix() const
    {
      const bool system_matrix_is_empty =
        system_matrix.m() == 0 || system_matrix.n() == 0;

      if (system_matrix_is_empty)
        {
          MyScope scope(this->timer, label + "::matrix::sp");

          system_matrix.clear();

          const auto &dof_handler =
            this->matrix_free.get_dof_handler(dof_index);

          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(),
            dof_handler.get_triangulation().get_communicator());
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
          dsp.compress();

          system_matrix.reinit(dsp);
        }

      {
        MyScope scope(this->timer, label + "::matrix::compute");

        if (system_matrix_is_empty == false)
          {
            system_matrix = 0.0; // clear existing content
          }

#define OPERATION(c, d)                                               \
  MatrixFreeTools::compute_matrix(matrix_free,                        \
                                  constraints,                        \
                                  system_matrix,                      \
                                  &OperatorBase::do_vmult_cell<c, d>, \
                                  this,                               \
                                  dof_index);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      return system_matrix;
    }

  protected:
    template <int n_comp, int n_grains>
    void
    do_vmult_cell(
      FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> &phi) const
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
      FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> phi(
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

    const unsigned int dof_index;
    const unsigned int components_number;

    const std::string label;

    mutable TrilinosWrappers::SparseMatrix system_matrix;

    ConditionalOStream  pcout;
    mutable TimerOutput timer;
  };



  template <typename Number, typename VectorizedArrayType>
  class ConstantsTracker
  {
  public:
    ConstantsTracker()
      : n_th(1)
      , max_level(2)
    {}

    void
    initialize(const unsigned int n_filled_lanes)
    {
      this->n_filled_lanes = n_filled_lanes;
      this->temp_min.clear();
      this->temp_max.clear();
    }

    void
    emplace_back(const unsigned int level, const Number &value)
    {
      if (level > max_level)
        return;

      temp_min.emplace_back(value);
      temp_max.emplace_back(value);
    }

    void
    emplace_back(const unsigned int level, const VectorizedArrayType &value)
    {
      if (level > max_level)
        return;

      const auto [min_value, max_value] = get_min_max(value);
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
    }

    template <int dim>
    void
    emplace_back(const unsigned int                         level,
                 const Tensor<1, dim, VectorizedArrayType> &value)
    {
      if (level > max_level)
        return;

#if false
      for (unsigned int d = 0; d < dim; ++d)
          {
            const auto [min_value, max_value] = get_min_max(value[d]);
            temp_min.emplace_back(min_value);
            temp_max.emplace_back(max_value);
          }
#else
      const auto [min_value, max_value] = get_min_max(value.norm());
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
#endif
    }

    template <int dim>
    void
    emplace_back(const unsigned int                         level,
                 const Tensor<2, dim, VectorizedArrayType> &value)
    {
      if (level > max_level)
        return;

#if false
      for (unsigned int d0 = 0; d0 < dim; ++d0)
        for (unsigned int d1 = 0; d1 < dim; ++d1)
            {
              const auto [min_value, max_value] = get_min_max(value[d0][d1]);
              temp_min.emplace_back(min_value);
              temp_max.emplace_back(max_value);
            }
#else
      const auto [min_value, max_value] = get_min_max(value.norm());
      temp_min.emplace_back(min_value);
      temp_max.emplace_back(max_value);
#endif
    }

    void
    finalize_point()
    {
      if (temp_min_0.size() == 0)
        temp_min_0 = temp_min;
      else
        {
          for (unsigned int i = 0; i < temp_min_0.size(); ++i)
            temp_min_0[i] = std::min(temp_min_0[i], temp_min[i]);
        }

      if (temp_max_0.size() == 0)
        temp_max_0 = temp_max;
      else
        {
          for (unsigned int i = 0; i < temp_max_0.size(); ++i)
            temp_max_0[i] = std::max(temp_max_0[i], temp_max[i]);
        }

      temp_min.clear();
      temp_max.clear();
    }

    void
    finalize()
    {
      std::vector<Number> global_min(temp_min_0.size());
      Utilities::MPI::min(temp_min_0, MPI_COMM_WORLD, global_min);
      all_values_min.emplace_back(global_min);

      std::vector<Number> global_max(temp_max_0.size());
      Utilities::MPI::max(temp_max_0, MPI_COMM_WORLD, global_max);
      all_values_max.emplace_back(global_max);

      this->temp_min_0.clear();
      this->temp_max_0.clear();
    }

    void
    print()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          const auto internal_print = [this](const auto &all_values,
                                             const auto &label) {
            std::ofstream pcout;
            pcout.open(label);

            unsigned int i = 0;

            for (; i < all_values.size() - 1; i += n_th)
              {
                for (unsigned int j = 0; j < all_values[i].size(); ++j)
                  pcout << all_values[i][j] << " ";

                pcout << std::endl;
              }

            if (n_th != 0)
              if ((i + 2) != (all_values.size() + n_th)) // print last entry
                {
                  i = all_values.size() - 2;

                  for (unsigned int j = 0; j < all_values[i].size(); ++j)
                    pcout << all_values[i][j] << " ";

                  pcout << std::endl;
                }

            pcout.close();
          };

          internal_print(all_values_min, "constants_min.txt");
          internal_print(all_values_max, "constants_max.txt");
        }
    }

  private:
    std::pair<Number, Number>
    get_min_max(const VectorizedArrayType &value) const
    {
      Number min_val = 0;
      Number max_val = 0;

      for (unsigned int i = 0; i < n_filled_lanes; ++i)
        {
          const auto val = value[i];

          if (i == 0)
            {
              min_val = val;
              max_val = val;
            }
          else
            {
              min_val = std::min(val, min_val);
              max_val = std::max(val, max_val);
            }
        }

      return {min_val, max_val};
    }

    unsigned int        n_filled_lanes;
    std::vector<Number> temp_min;
    std::vector<Number> temp_max;

    std::vector<Number> temp_min_0;
    std::vector<Number> temp_max_0;

    std::vector<std::vector<Number>> all_values_min;
    std::vector<std::vector<Number>> all_values_max;

    const unsigned int n_th;
    const unsigned int max_level;
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
  };



  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperator
    : public OperatorBase<dim,
                          Number,
                          VectorizedArrayType,
                          SinteringOperator<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = OperatorBase<dim,
                           Number,
                           VectorizedArrayType,
                           SinteringOperator<dim, Number, VectorizedArrayType>>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperator(
      const MatrixFree<dim, Number, VectorizedArrayType> &   matrix_free,
      const std::vector<const AffineConstraints<Number> *> & constraints,
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
    {
#ifdef WITH_TRACKER
      tracker.print();
#endif
    }

    void
    evaluate_nonlinear_residual(VectorType &dst, const VectorType &src) const
    {
      MyScope scope(this->timer, "sintering_op::nonlinear_residual");

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
    set_previous_solution(const VectorType &src) const
    {
      this->old_solution = src;
      this->old_solution.update_ghost_values();
    }

    const VectorType &
    get_previous_solution() const
    {
      this->old_solution.zero_out_ghost_values();
      return this->old_solution;
    }

    void
    evaluate_newton_step(const VectorType &newton_step)
    {
      MyScope scope(this->timer, "sintering_op::newton_step");

      const unsigned n_cells = this->matrix_free.n_cell_batches();
      const unsigned n_quadrature_points =
        this->matrix_free.get_quadrature().size();

      nonlinear_values.reinit(
        {n_cells, n_quadrature_points, this->n_components()});
      nonlinear_gradients.reinit(
        {n_cells, n_quadrature_points, this->n_components()});

      int dummy = 0;

#define OPERATION(c, d)                                \
  this->matrix_free.cell_loop(                         \
    &SinteringOperator::do_evaluate_newton_step<c, d>, \
    this,                                              \
    dummy,                                             \
    newton_step);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

#ifdef WITH_TRACKER
      tracker.finalize();
#endif

      this->newton_step = newton_step;
      this->newton_step.update_ghost_values();

      if (matrix_based)
        this->get_system_matrix(); // assemble matrix
    }

    void
    set_timestep(double dt_new)
    {
      this->dt = dt_new;
    }

    const SinteringOperatorData<dim, VectorizedArrayType> &
    get_data() const
    {
      return data;
    }

    const double &
    get_dt() const
    {
      return this->dt;
    }

    const Table<3, VectorizedArrayType> &
    get_nonlinear_values() const
    {
      return nonlinear_values;
    }


    const Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> &
    get_nonlinear_gradients() const
    {
      return nonlinear_gradients;
    }

    template <int n_comp, int n_grains>
    void
    do_add_data_vectors(DataOut<dim> &data_out, const VectorType &vec) const
    {
      if constexpr (n_comp <= 2)
        {
          Assert(false, ExcNotImplemented());
          (void)data_out;
          (void)vec;
        }
      else
        {
          constexpr unsigned int n_entries =
            8 + 3 * n_grains + n_grains * (n_grains - 1) / 2;
          std::array<VectorType, n_entries> data_vectors;

          for (auto &data_vector : data_vectors)
            this->matrix_free.initialize_dof_vector(data_vector, 3);

          FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType>
            fe_eval_all(this->matrix_free);
          FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> fe_eval(
            this->matrix_free, 3 /*scalar dof index*/);

          MatrixFreeOperators::
            CellwiseInverseMassMatrix<dim, -1, 1, Number, VectorizedArrayType>
              inverse_mass_matrix(fe_eval);

          AlignedVector<VectorizedArrayType> buffer(fe_eval.n_q_points *
                                                    n_entries);

          const auto &free_energy = this->data.free_energy;
          const auto &L           = this->data.L;
          const auto &mobility    = this->data.mobility;
          const auto &kappa_c     = this->data.kappa_c;
          const auto &kappa_p     = this->data.kappa_p;
          const auto  dt_inv      = 1.0 / dt;

          vec.update_ghost_values();

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
                  std::array<const Tensor<1, dim, VectorizedArrayType> *,
                             n_grains>
                    etas_grad;

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      etas[ig]      = &val[2 + ig];
                      etas_grad[ig] = &grad[2 + ig];
                    }

                  std::array<VectorizedArrayType, n_entries> temp;

                  unsigned int counter = 0;

                  temp[counter++] = VectorizedArrayType(dt_inv);
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

                  temp[counter++] = mobility.M(c, etas, c_grad, etas_grad);
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

                  temp[counter++] = VectorizedArrayType(kappa_c);
                  temp[counter++] = VectorizedArrayType(kappa_p);
                  temp[counter++] = VectorizedArrayType(L);

                  for (unsigned int c = 0; c < n_entries; ++c)
                    buffer[c * fe_eval.n_q_points + q] = temp[c];
                }

              for (unsigned int c = 0; c < n_entries; ++c)
                {
                  inverse_mass_matrix.transform_from_q_points_to_basis(
                    1,
                    buffer.data() + c * fe_eval.n_q_points,
                    fe_eval.begin_dof_values());

                  fe_eval.set_dof_values(data_vectors[c]);
                }
            }

          vec.zero_out_ghost_values();

          for (unsigned int c = 0; c < n_entries; ++c)
            {
              std::ostringstream ss;
              ss << "aux_" << std::setw(2) << std::setfill('0') << c;

              data_out.add_data_vector(this->matrix_free.get_dof_handler(3),
                                       data_vectors[c],
                                       ss.str());
            }
        }
    }

    void
    add_data_vectors(DataOut<dim> &data_out, const VectorType &vec) const
    {
#define OPERATION(c, d) this->do_add_data_vectors<c, d>(data_out, vec);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
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
      FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> &phi) const
    {
      const unsigned int cell = phi.get_current_cell_index();

      const auto &free_energy = this->data.free_energy;
      const auto &L           = this->data.L;
      const auto &mobility    = this->data.mobility;
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  dt_inv      = 1.0 / dt;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto val  = nonlinear_values[cell][q];
          const auto grad = nonlinear_gradients[cell][q];

          const auto c       = val[0];
          const auto c_grad  = grad[0];
          const auto mu_grad = grad[1];

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
      VectorType &                                        dst,
      const VectorType &                                  src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      if constexpr (n_comp <= 2)
        {
          Assert(false, ExcNotImplemented());
          (void)matrix_free;
          (void)dst;
          (void)src;
          (void)range;
        }
      else
        {
          FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> phi_old(
            matrix_free);
          FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> phi(
            matrix_free);

          const auto &free_energy = this->data.free_energy;
          const auto &L           = this->data.L;
          const auto &mobility    = this->data.mobility;
          const auto &kappa_c     = this->data.kappa_c;
          const auto &kappa_p     = this->data.kappa_p;

          for (auto cell = range.first; cell < range.second; ++cell)
            {
              phi_old.reinit(cell);
              phi.reinit(cell);

              phi.gather_evaluate(
                src,
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
                  std::array<const Tensor<1, dim, VectorizedArrayType> *,
                             n_grains>
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
                  value_result[0] = (c - c_old) / dt;
                  value_result[1] = -mu + free_energy.df_dc(c, etas);
                  gradient_result[0] =
                    mobility.M(c, etas, c_grad, etas_grad) * grad[1];
                  gradient_result[1] = kappa_c * grad[0];

                  // AC equations
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      value_result[2 + ig] =
                        (val[2 + ig] - val_old[2 + ig]) / dt +
                        L * free_energy.df_detai(c, etas, ig);

                      gradient_result[2 + ig] = L * kappa_p * grad[2 + ig];
                    }

                  phi.submit_value(value_result, q);
                  phi.submit_gradient(gradient_result, q);
                }
              phi.integrate_scatter(
                EvaluationFlags::EvaluationFlags::values |
                  EvaluationFlags::EvaluationFlags::gradients,
                dst);
            }
        }
    }

    template <int n_comp, int n_grains>
    void
    do_evaluate_newton_step(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      int &,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &range)
    {
      if constexpr (n_comp <= 2)
        {
          Assert(false, ExcNotImplemented());
          (void)matrix_free;
          (void)src;
          (void)range;
        }
      else
        {
#ifdef WITH_TRACKER
          const auto &free_energy = this->data.free_energy;
          const auto &L           = this->data.L;
          const auto &mobility    = this->data.mobility;
          const auto &kappa_c     = this->data.kappa_c;
          const auto &kappa_p     = this->data.kappa_p;
          const auto  dt_inv      = 1.0 / dt;
#endif

          FEEvaluation<dim, -1, 0, n_comp, Number, VectorizedArrayType> phi(
            matrix_free);

          for (auto cell = range.first; cell < range.second; ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values_plain(src);
              phi.evaluate(EvaluationFlags::values |
                           EvaluationFlags::gradients);

#ifdef WITH_TRACKER
              tracker.initialize(
                matrix_free.n_active_entries_per_cell_batch(cell));
#endif

              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  for (unsigned int c = 0; c < n_comp; ++c)
                    {
                      nonlinear_values(cell, q, c)    = phi.get_value(q)[c];
                      nonlinear_gradients(cell, q, c) = phi.get_gradient(q)[c];
                    }

#ifdef WITH_TRACKER
                  const auto val  = nonlinear_values[cell][q];
                  const auto grad = nonlinear_gradients[cell][q];

                  const auto c       = val[0];
                  const auto c_grad  = grad[0];
                  const auto mu_grad = grad[1];

                  std::array<const VectorizedArrayType *, n_grains> etas;
                  std::array<const Tensor<1, dim, VectorizedArrayType> *,
                             n_grains>
                    etas_grad;

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      etas[ig]      = &val[2 + ig];
                      etas_grad[ig] = &grad[2 + ig];
                    }

                  tracker.emplace_back(0, dt_inv);
                  tracker.emplace_back(1, free_energy.d2f_dc2(c, etas));
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      tracker.emplace_back(
                        2, free_energy.d2f_dcdetai(c, etas, ig));
                    }
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      tracker.emplace_back(2,
                                           free_energy.d2f_detai2(c, etas, ig));
                    }
                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                        {
                          tracker.emplace_back(
                            2, free_energy.d2f_detaidetaj(c, etas, ig, jg));
                        }
                    }
                  tracker.emplace_back(0,
                                       mobility.M(c, etas, c_grad, etas_grad));
                  tracker.emplace_back(
                    1, mobility.dM_dc(c, etas, c_grad, etas_grad) * mu_grad);
                  tracker.emplace_back(1,
                                       mobility.dM_dgrad_c(c, c_grad, mu_grad));

                  for (unsigned int ig = 0; ig < n_grains; ++ig)
                    {
                      tracker.emplace_back(
                        2,
                        mobility.dM_detai(c, etas, c_grad, etas_grad, ig) *
                          mu_grad);
                    }
                  tracker.emplace_back(0, kappa_c);
                  tracker.emplace_back(0, kappa_p);
                  tracker.emplace_back(0, L);

                  tracker.finalize_point();
#endif
                }
            }
        }
    }

    SinteringOperatorData<dim, VectorizedArrayType> data;

    double dt;

    mutable VectorType old_solution, newton_step;

    Table<3, VectorizedArrayType>                         nonlinear_values;
    Table<3, dealii::Tensor<1, dim, VectorizedArrayType>> nonlinear_gradients;

    ConstantsTracker<Number, VectorizedArrayType> tracker;

    const bool matrix_based;
  };

} // namespace Sintering
