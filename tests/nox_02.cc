#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <NOX_Abstract_Vector.H>

using namespace dealii;

namespace dealii
{
  namespace internal
  {
    namespace NOXWrapper
    {
      template <typename VectorType>
      class Vector : public NOX::Abstract::Vector
      {
        NOX::Abstract::Vector &
        init(double gamma) override
        {
          AssertThrow(false, ExcNotImplemented());
          (void)gamma;

          return *this;
        }

        NOX::Abstract::Vector &
        random(bool useSeed = false, int seed = 1) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)useSeed;
          (void)seed;

          return *this;
        }

        NOX::Abstract::Vector &
        abs(const NOX::Abstract::Vector &y) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)y;

          return *this;
        }

        NOX::Abstract::Vector &
        operator=(const NOX::Abstract::Vector &y) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)y;

          return *this;
        }

        NOX::Abstract::Vector &
        reciprocal(const NOX::Abstract::Vector &y) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)y;

          return *this;
        }

        NOX::Abstract::Vector &
        scale(double gamma) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)gamma;

          return *this;
        }

        NOX::Abstract::Vector &
        scale(const NOX::Abstract::Vector &a) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)a;

          return *this;
        }

        NOX::Abstract::Vector &
        update(double                       alpha,
               const NOX::Abstract::Vector &a,
               double                       gamma = 0.0) override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)alpha;
          (void)a;
          (void)gamma;

          return *this;
        }

        NOX::Abstract::Vector &
        update(double                       alpha,
               const NOX::Abstract::Vector &a,
               double                       beta,
               const NOX::Abstract::Vector &b,
               double                       gamma)
        {
          AssertThrow(false, ExcNotImplemented());

          (void)alpha;
          (void)a;
          (void)beta;
          (void)b;
          (void)gamma;

          return *this;
        }

        Teuchos::RCP<NOX::Abstract::Vector> clone(NOX::CopyType) const override
        {
          AssertThrow(false, ExcNotImplemented());

          return {};
        }

        double
        norm(NOX::Abstract::Vector::NormType type =
               NOX::Abstract::Vector::TwoNorm) const
        {
          AssertThrow(false, ExcNotImplemented());
          (void)type;

          return 0.0;
        }

        double
        norm(const NOX::Abstract::Vector &weights) const override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)weights;

          return 0.0;
        }

        double
        innerProduct(const NOX::Abstract::Vector &y) const override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)y;

          return 0.0;
        }

        NOX::size_type
        length() const override
        {
          AssertThrow(false, ExcNotImplemented());

          return 0.0;
        }

      private:
        std::shared_ptr<VectorType> vector;
      };



    } // namespace NOXWrapper


  } // namespace internal

} // namespace dealii

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  internal::NOXWrapper::Vector<VectorType> vector;
}