#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Solver_Factory.H>
#include <NOX_Solver_Generic.H>
#include <NOX_StatusTest_Combo.H>
#include <NOX_StatusTest_MaxIters.H>
#include <NOX_StatusTest_NormF.H>

using namespace dealii;

namespace dealii
{
  namespace internal
  {
    namespace NOXWrapper
    {
      template <typename VectorType>
      class Group;

      template <typename VectorType>
      class Vector : public NOX::Abstract::Vector
      {
      public:
        Vector() = default;

        Vector(VectorType &vector)
        {
          this->vector.reset(&vector, [](auto *) { /*nothing to do*/ });
        }

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
          if (vector == nullptr)
            vector = std::shared_ptr<VectorType>();

          const auto y_ = dynamic_cast<const Vector<VectorType> *>(&y);

          Assert(y_, ExcInternalError());

          vector->reinit(*y_->vector);

          *vector = *y_->vector;

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

        Teuchos::RCP<NOX::Abstract::Vector>
        clone(NOX::CopyType copy_type) const override
        {
          auto new_vector    = Teuchos::rcp(new Vector<VectorType>());
          new_vector->vector = std::make_shared<VectorType>();
          new_vector->vector->reinit(*this->vector);

          if (copy_type == NOX::CopyType::DeepCopy)
            *new_vector->vector = *this->vector;

          return new_vector;
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

        friend Group<VectorType>;
      };

      template <typename VectorType>
      class Group : public NOX::Abstract::Group
      {
      public:
        Group(
          VectorType &                                                 solution,
          const std::function<void(const VectorType &, VectorType &)> &residual,
          const std::function<void(const VectorType &, const bool)>
            &setup_jacobian,
          const std::function<unsigned int(const VectorType &, VectorType &)>
            &solve_with_jacobian)
          : x(solution)
          , residual(residual)
          , setup_jacobian(setup_jacobian)
          , solve_with_jacobian(solve_with_jacobian)
          , is_valid_f(false)
          , is_valid_j(false)
        {}

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
          if (is_valid_f == false)
            {
              f.vector = std::make_shared<VectorType>();
              f.vector->reinit(*x.vector);

              residual(*x.vector, *f.vector);
              is_valid_f = true;
            }

          return NOX::Abstract::Group::Ok;
        }

        bool
        isF() const override
        {
          return is_valid_f;
        }

        NOX::Abstract::Group::ReturnType
        computeJacobian() override
        {
          if (is_valid_j == false)
            {
              setup_jacobian(*x.vector, true);

              is_valid_j = true;
            }

          return NOX::Abstract::Group::Ok;
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

        Teuchos::RCP<NOX::Abstract::Group>
        clone(NOX::CopyType copy_type) const override
        {
          auto new_group = Teuchos::rcp(new Group<VectorType>(
            *x.vector, residual, setup_jacobian, solve_with_jacobian));

          if (copy_type == NOX::CopyType::DeepCopy)
            new_group->is_valid_f = is_valid_f;

          return new_group;
        }

        NOX::Abstract::Group::ReturnType
        computeNewton(Teuchos::ParameterList &p)
        {
          AssertThrow(false, ExcNotImplemented());

          (void)p;

          return {};
        }

        NOX::Abstract::Group::ReturnType
        applyJacobianInverse(Teuchos::ParameterList &     params,
                             const NOX::Abstract::Vector &input,
                             NOX::Abstract::Vector &      result) const override
        {
          AssertThrow(false, ExcNotImplemented());

          (void)params;
          (void)input;
          (void)result;

          return {};
        }

      private:
        Vector<VectorType> x, f, gradient, newton;

        std::function<void(const VectorType &, VectorType &)> residual;
        std::function<void(const VectorType &, const bool)>   setup_jacobian;
        std::function<unsigned int(const VectorType &, VectorType &)>
          solve_with_jacobian;

        bool is_valid_f;
        bool is_valid_j;
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

  // some parameters
  const unsigned int n_max_iterations = 100;
  const double       abs_tolerance    = 1e-9;

  std::function<void(const VectorType &, VectorType &)> residual =
    [](const auto &, auto &) {};
  std::function<void(const VectorType &, const bool)> setup_jacobian =
    [](const auto &, const auto) {};
  std::function<unsigned int(const VectorType &, VectorType &)>
    solve_with_jacobian = [](const auto &, auto &) { return 0; };

  // initial guess
  VectorType solution;

  // create group
  const auto group = Teuchos::rcp(new internal::NOXWrapper::Group<VectorType>(
    solution, residual, setup_jacobian, solve_with_jacobian));

  // setup parameters
  Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
    Teuchos::rcp(new Teuchos::ParameterList);

  non_linear_parameters->set("Nonlinear Solver", "Line Search Based");

  auto &dir_parameters = non_linear_parameters->sublist("Direction");
  dir_parameters.set("Method", "Newton");

  auto &search_parameters = non_linear_parameters->sublist("Line Search");
  search_parameters.set("Method", "Full Step");

  // setup solver control
  const auto solver_control_norm_f =
    Teuchos::rcp(new NOX::StatusTest::NormF(abs_tolerance));

  const auto solver_control_max_iterations =
    Teuchos::rcp(new NOX::StatusTest::MaxIters(n_max_iterations));

  const auto combo =
    Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR,
                                            solver_control_norm_f,
                                            solver_control_max_iterations));

  // create non-linear solver
  const auto solver =
    NOX::Solver::buildSolver(group, combo, non_linear_parameters);

  // solve
  const auto status = solver->solve();
  AssertThrow(status == NOX::StatusTest::Converged, ExcNotImplemented());
}