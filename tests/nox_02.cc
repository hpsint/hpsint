#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <NOX_Abstract_Group.H>
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
      public:
        NOX::Abstract::Vector &
        init(double gamma) override
        {
          *vector = gamma;
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
          *vector *= gamma;

          return *this;
        }

        NOX::Abstract::Vector &
        scale(const NOX::Abstract::Vector &a) override
        {
          const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);

          Assert(a_, ExcInternalError());

          vector->scale(*a_->vector);

          return *this;
        }

        NOX::Abstract::Vector &
        update(double                       alpha,
               const NOX::Abstract::Vector &a,
               double                       gamma = 0.0) override
        {
          const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);

          Assert(a_, ExcInternalError());

          vector->sadd(gamma, alpha, *a_->vector);

          return *this;
        }

        NOX::Abstract::Vector &
        update(double                       alpha,
               const NOX::Abstract::Vector &a,
               double                       beta,
               const NOX::Abstract::Vector &b,
               double                       gamma)
        {
          const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);
          const auto b_ = dynamic_cast<const Vector<VectorType> *>(&b);

          Assert(a_, ExcInternalError());
          Assert(b_, ExcInternalError());

          vector->operator*=(gamma);
          vector->add(alpha, *a_->vector, beta, *b_->vector);

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
          if (type == NOX::Abstract::Vector::NormType::TwoNorm)
            return vector->l2_norm();
          if (type == NOX::Abstract::Vector::NormType::OneNorm)
            return vector->l1_norm();
          if (type == NOX::Abstract::Vector::NormType::MaxNorm)
            return vector->linfty_norm();

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
          const auto y_ = dynamic_cast<const Vector<VectorType> *>(&y);

          Assert(y_, ExcInternalError());

          return (*vector) * (*y_->vector);
        }

        NOX::size_type
        length() const override
        {
          return vector->size();
        }

      private:
        std::shared_ptr<VectorType> vector;
      };

      template <typename VectorType>
      class Group : public NOX::Abstract::Group
      {
      public:
        NOX::Abstract::Group &
        operator=(const NOX::Abstract::Group &source) override
        {
          AssertThrow(false, ExcNotImplemented());
          (void)source;

          return *this;
        }

        void
        setX(const NOX::Abstract::Vector &y) override
        {
          x = y;
        }

        void
        computeX(const NOX::Abstract::Group & grp,
                 const NOX::Abstract::Vector &d,
                 double                       step)
        {
          AssertThrow(false, ExcNotImplemented());
          (void)grp;
          (void)d;
          (void)step;
        }

        NOX::Abstract::Group::ReturnType
        computeF() override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        bool
        isF() const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        const NOX::Abstract::Vector &
        getX() const override
        {
          return x;
        }

        const NOX::Abstract::Vector &
        getF() const override
        {
          return f;
        }

        double
        getNormF() const override
        {
          return f.norm();
        }

        const NOX::Abstract::Vector &
        getGradient() const override
        {
          return gradient;
        }

        const NOX::Abstract::Vector &
        getNewton() const override
        {
          return newton;
        }

        Teuchos::RCP<const NOX::Abstract::Vector>
        getXPtr() const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        Teuchos::RCP<const NOX::Abstract::Vector>
        getFPtr() const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        Teuchos::RCP<const NOX::Abstract::Vector>
        getGradientPtr() const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        Teuchos::RCP<const NOX::Abstract::Vector>
        getNewtonPtr() const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

        Teuchos::RCP<NOX::Abstract::Group> clone(NOX::CopyType) const override
        {
          AssertThrow(false, ExcNotImplemented());
          return {};
        }

      private:
        Vector<VectorType> x, f, gradient, newton;
      };



    } // namespace NOXWrapper
  }   // namespace internal
} // namespace dealii

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  internal::NOXWrapper::Vector<VectorType> vector;

  internal::NOXWrapper::Group<VectorType> group;
}