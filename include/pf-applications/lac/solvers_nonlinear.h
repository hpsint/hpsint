#pragma once

#include <deal.II/lac/vector.h>

#include <pf-applications/base/timer.h>

namespace NonLinearSolvers
{
  using namespace dealii;

  template <typename VectorType>
  class NewtonSolver;

  template <typename VectorType>
  class NOXSolver;


  struct NewtonSolverSolverControl
  {
  public:
    enum State
    {
      iterate,
      success,
      failure
    };

    NewtonSolverSolverControl(const unsigned int max_iter = 10,
                              const double       abs_tol  = 1.e-20,
                              const double       rel_tol  = 1.e-5)
      : max_iter(max_iter)
      , abs_tol(abs_tol)
      , rel_tol(rel_tol)
    {}

    void
    clear()
    {
      newton_iterations    = 0;
      linear_iterations    = 0;
      residual_evaluations = 0;
    }

    unsigned int
    n_newton_iterations() const
    {
      return newton_iterations;
    }

    unsigned int
    n_linear_iterations() const
    {
      return linear_iterations;
    }

    unsigned int
    n_residual_evaluations() const
    {
      return residual_evaluations;
    }

    void
    increment_newton_iterations(const unsigned int num)
    {
      newton_iterations += num;
    }

    void
    increment_linear_iterations(const unsigned int num)
    {
      linear_iterations += num;
    }

    void
    increment_residual_evaluations(const unsigned int num)
    {
      residual_evaluations += num;
    }

    template <typename VectorType>
    State
    check(const unsigned int step,
          const double       check_value,
          const VectorType & solution,
          const VectorType & residuum)
    {
      (void)solution;
      (void)residuum;

      if (step == 0)
        this->check_value_0 = check_value;

      this->check_value = check_value;

      this->newton_iterations = step;

      if (check_value > abs_tol && check_value / check_value_0 > rel_tol &&
          newton_iterations < max_iter)
        return iterate;

      if (check_value <= abs_tol || check_value / check_value_0 <= rel_tol)
        return success;

      return failure;
    }

    State
    check()
    {
      Vector<double> dummy;

      return check(this->newton_iterations, this->check_value, dummy, dummy);
    }

    unsigned int
    get_max_iter() const
    {
      return max_iter;
    }

    double
    get_abs_tol() const
    {
      return abs_tol;
    }

    double
    get_rel_tol() const
    {
      return rel_tol;
    }

  private:
    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;

    double check_value_0 = 0.0;
    double check_value   = 0.0;

    unsigned int newton_iterations    = 0;
    unsigned int linear_iterations    = 0;
    unsigned int residual_evaluations = 0;
  };



  struct NewtonSolverAdditionalData
  {
    NewtonSolverAdditionalData(const bool         do_update             = true,
                               const unsigned int threshold_newton_iter = 10,
                               const unsigned int threshold_linear_iter = 20,
                               const bool         reuse_preconditioner  = true,
                               const bool         use_damping           = true)
      : do_update(do_update)
      , threshold_newton_iter(threshold_newton_iter)
      , threshold_linear_iter(threshold_linear_iter)
      , reuse_preconditioner(reuse_preconditioner)
      , use_damping(use_damping)
    {}

    const bool         do_update;
    const unsigned int threshold_newton_iter;
    const unsigned int threshold_linear_iter;
    const bool         reuse_preconditioner;
    const bool         use_damping;
  };



  class ExcNewtonDidNotConverge : public dealii::ExceptionBase
  {
  public:
    ExcNewtonDidNotConverge(const std::string &label)
      : label(label)
    {}

    virtual ~ExcNewtonDidNotConverge() noexcept override = default;

    virtual void
    print_info(std::ostream &out) const override
    {
      out << message() << std::endl;
    }

    std::string
    message() const
    {
      return "Maximum number of " + this->label +
             " iterations of damp Netwon exceeded!";
    }

  private:
    const std::string label;
  };



  template <typename VectorType>
  class NewtonSolver
  {
  public:
    virtual void
    solve(VectorType &dst) const = 0;

    virtual void
    clear() const = 0;

    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
    std::function<NewtonSolverSolverControl::State(const unsigned int,
                                                   const double,
                                                   const VectorType &,
                                                   const VectorType &)>
      check_iteration_status = {};
  };



  template <typename VectorType>
  class DampedNewtonSolver : public NewtonSolver<VectorType>
  {
  public:
    DampedNewtonSolver(NewtonSolverSolverControl &       statistics,
                       const NewtonSolverAdditionalData &solver_data_in =
                         NewtonSolverAdditionalData())
      : statistics(statistics)
      , solver_data(solver_data_in)
    {}

    void
    solve(VectorType &dst) const override
    {
      VectorType vec_residual, increment, tmp;
      this->reinit_vector(vec_residual);
      this->reinit_vector(increment);
      this->reinit_vector(tmp);

      if (this->solver_data.reuse_preconditioner == false)
        clear();

      // Accumulated linear iterations
      this->statistics.clear();

      // evaluate residual using the given estimate of the solution
      this->residual(dst, vec_residual);

      double   norm_r = vec_residual.l2_norm();
      unsigned it     = 0;

      auto status = check(it, norm_r, dst, vec_residual);

      while (status == NewtonSolverSolverControl::iterate)
        {
          // reset increment
          increment = 0.0;

          // multiply by -1.0 since the linearized problem is "LinearMatrix *
          // increment = - vec_residual"
          vec_residual *= -1.0;

          // solve linear problem
          const bool threshold_exceeded =
            (history_newton_iterations % solver_data.threshold_newton_iter ==
             0) ||
            (history_linear_iterations_last >
             solver_data.threshold_linear_iter);

          this->setup_jacobian(dst,
                               solver_data.do_update && threshold_exceeded);

          history_linear_iterations_last =
            this->solve_with_jacobian(vec_residual, increment);

          if (this->solver_data.use_damping)
            {
              // damped Newton scheme
              const double tau =
                0.1; // another parameter (has to be smaller than 1)
              const unsigned int N_ITER_TMP_MAX =
                100;              // iteration counts for damping scheme
              double omega = 1.0; // damping factor
              double norm_r_tmp =
                1.0; // norm of residual using temporary solution
              unsigned int n_iter_tmp = 0;

              do
                {
                  // calculate temporary solution
                  tmp = dst;
                  tmp.add(omega, increment);

                  // evaluate residual using the temporary solution
                  this->residual(tmp, vec_residual);

                  // calculate norm of residual (for temporary solution)
                  norm_r_tmp = vec_residual.l2_norm();

                  // reduce step length
                  omega = omega / 2.0;

                  // increment counter
                  n_iter_tmp++;
                }
              while (norm_r_tmp >= (1.0 - tau * omega) * norm_r &&
                     n_iter_tmp < N_ITER_TMP_MAX);

              AssertThrow(norm_r_tmp < (1.0 - tau * omega) * norm_r,
                          ExcNewtonDidNotConverge("damping"));

              // update solution and residual
              dst    = tmp;
              norm_r = norm_r_tmp;
            }
          else
            {
              // non-damped Newton scheme

              // calculate temporary solution
              dst.add(1.0, increment);

              // evaluate residual
              this->residual(dst, vec_residual);

              // calculate norm of residual
              norm_r = vec_residual.l2_norm();
            }

          // increment iteration counter
          ++it;
          ++history_newton_iterations;

          status = check(it, norm_r, dst, vec_residual);
        }

      AssertThrow(status == NewtonSolverSolverControl::success,
                  ExcNewtonDidNotConverge("Newton"));
    }

    void
    clear() const override
    {
      history_linear_iterations_last = 0;
      history_newton_iterations      = 0;
    }

  private:
    NewtonSolverSolverControl::State
    check(const unsigned int step,
          const double       check_value,
          const VectorType & x,
          const VectorType & r) const
    {
      if (this->check_iteration_status == nullptr)
        return statistics.check(step, check_value, x, r);

      const auto state1 = statistics.check(step, check_value, x, r);
      const auto state2 = this->check_iteration_status(step, check_value, x, r);

      if ((state1 == NewtonSolverSolverControl::failure) ||
          (state2 == NewtonSolverSolverControl::failure))
        return NewtonSolverSolverControl::failure;
      else if ((state1 == NewtonSolverSolverControl::iterate) ||
               (state2 == NewtonSolverSolverControl::iterate))
        return NewtonSolverSolverControl::iterate;
      else
        return NewtonSolverSolverControl::success;
    }

    NewtonSolverSolverControl &      statistics;
    const NewtonSolverAdditionalData solver_data;

    mutable unsigned int history_linear_iterations_last = 0;
    mutable unsigned int history_newton_iterations      = 0;
  };
} // namespace NonLinearSolvers



#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Solver_Factory.H>
#include <NOX_Solver_Generic.H>
#include <NOX_StatusTest_Combo.H>
#include <NOX_StatusTest_MaxIters.H>
#include <NOX_StatusTest_NormF.H>
#include <NOX_StatusTest_RelativeNormF.H>



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
               double                       gamma = 0.0)
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

        std::shared_ptr<VectorType>
        genericVector() const
        {
          return vector;
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
          if (this != &source)
            {
              const auto other =
                dynamic_cast<const Group<VectorType> *>(&source);

              Assert(other, ExcInternalError());

              if (other->x.vector)
                {
                  if (this->x.vector == nullptr)
                    this->x.vector = std::make_shared<VectorType>();

                  *this->x.vector = *other->x.vector;
                }
              else
                {
                  this->x.vector = {};
                }

              if (other->f.vector)
                {
                  if (this->f.vector == nullptr)
                    this->f.vector = std::make_shared<VectorType>();

                  *this->f.vector = *other->x.vector;
                }
              else
                {
                  this->f.vector = {};
                }

              if (other->gradient.vector)
                {
                  if (this->gradient.vector == nullptr)
                    this->gradient.vector = std::make_shared<VectorType>();

                  *this->gradient.vector = *other->gradient.vector;
                }
              else
                {
                  this->gradient.vector = {};
                }

              if (other->newton.vector)
                {
                  if (this->newton.vector == nullptr)
                    this->newton.vector = std::make_shared<VectorType>();

                  *this->newton.vector = *other->newton.vector;
                }
              else
                {
                  this->newton.vector = {};
                }

              this->residual            = other->residual;
              this->setup_jacobian      = other->setup_jacobian;
              this->solve_with_jacobian = other->solve_with_jacobian;

              this->is_valid_f = other->is_valid_f;
              this->is_valid_j = other->is_valid_j;
            }

          return *this;
        }

        void
        setX(const NOX::Abstract::Vector &y) override
        {
          reset();

          x = y;
        }

        void
        computeX(const NOX::Abstract::Group & grp,
                 const NOX::Abstract::Vector &d,
                 double                       step)
        {
          reset();

          const auto grp_ = dynamic_cast<const Group *>(&grp);

          Assert(grp_, ExcInternalError());

          x.update(1.0, grp_->x, step, d);
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

        bool
        isJacobian() const override
        {
          return is_valid_j;
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

          if (x.vector)
            {
              new_group->x.vector = std::make_shared<VectorType>();
              new_group->x.vector->reinit(*x.vector);
            }

          if (f.vector)
            {
              new_group->f.vector = std::make_shared<VectorType>();
              new_group->f.vector->reinit(*f.vector);
            }

          if (gradient.vector)
            {
              new_group->gradient.vector = std::make_shared<VectorType>();
              new_group->gradient.vector->reinit(*gradient.vector);
            }

          if (newton.vector)
            {
              new_group->newton.vector = std::make_shared<VectorType>();
              new_group->newton.vector->reinit(*newton.vector);
            }

          if (copy_type == NOX::CopyType::DeepCopy)
            {
              if (x.vector)
                *new_group->x.vector = *x.vector;

              if (f.vector)
                *new_group->f.vector = *f.vector;

              if (gradient.vector)
                *new_group->gradient.vector = *gradient.vector;

              if (newton.vector)
                *new_group->newton.vector = *newton.vector;

              new_group->is_valid_f = is_valid_f;
              new_group->is_valid_j = is_valid_j;
            }

          return new_group;
        }

        NOX::Abstract::Group::ReturnType
        computeNewton(Teuchos::ParameterList &p)
        {
          (void)p; // TODO

          if (isNewton())
            return NOX::Abstract::Group::Ok;

          Assert(isF(), ExcMessage("Residual has not been computed yet!"));
          Assert(isJacobian(), ExcMessage("Jacobian has not been setup yet!"));

          if (newton.vector == nullptr)
            newton.vector = std::make_shared<VectorType>();

          newton.vector->reinit(*f.vector, false);

          solve_with_jacobian(*f.vector, *newton.vector);

          // TODO: use status of linear solver

          newton.scale(-1.0);

          return NOX::Abstract::Group::Ok;
        }

        NOX::Abstract::Group::ReturnType
        applyJacobian(const NOX::Abstract::Vector &input,
                      NOX::Abstract::Vector &      result) const override
        {
          if (!isJacobian())
            return NOX::Abstract::Group::BadDependency;

          const auto *input_ = dynamic_cast<const Vector<VectorType> *>(&input);
          const auto *result_ =
            dynamic_cast<const Vector<VectorType> *>(&result);

          solve_with_jacobian(*input_->vector, *result_->vector);

          return NOX::Abstract::Group::Ok;
        }

      private:
        void
        reset()
        {
          is_valid_f = false;
          is_valid_j = false;
        }

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

namespace NonLinearSolvers
{
  template <typename VectorType>
  class NOXCheck : public NOX::StatusTest::Generic
  {
  public:
    NOXCheck(std::function<NewtonSolverSolverControl::State(const unsigned int,
                                                            const double,
                                                            const VectorType &,
                                                            const VectorType &)>
                  check_iteration_status,
             bool as_dummy = false)
      : check_iteration_status(check_iteration_status)
      , as_dummy(as_dummy)
      , status(NOX::StatusTest::Unevaluated)
    {}

    NOX::StatusTest::StatusType
    checkStatus(const NOX::Solver::Generic &problem,
                NOX::StatusTest::CheckType  checkType) override
    {
      if (checkType == NOX::StatusTest::None)
        {
          status = NOX::StatusTest::Unevaluated;
        }
      else
        {
          if (check_iteration_status == nullptr)
            {
              status = NOX::StatusTest::Converged;
            }
          else
            {
              const auto &x = problem.getSolutionGroup().getX();
              const auto *x_ =
                dynamic_cast<const internal::NOXWrapper::Vector<VectorType> *>(
                  &x);

              const auto &f = problem.getSolutionGroup().getF();
              const auto *f_ =
                dynamic_cast<const internal::NOXWrapper::Vector<VectorType> *>(
                  &f);

              const unsigned int step = problem.getNumIterations();

              const double norm_f = f_->genericVector()->l2_norm();

              state = this->check_iteration_status(step,
                                                   norm_f,
                                                   *x_->genericVector(),
                                                   *f_->genericVector());

              switch (state)
                {
                  case NewtonSolverSolverControl::iterate:
                    status = NOX::StatusTest::Unconverged;
                    break;
                  case NewtonSolverSolverControl::failure:
                    status = NOX::StatusTest::Failed;
                    break;
                  case NewtonSolverSolverControl::success:
                    status = NOX::StatusTest::Converged;
                    break;
                  default:
                    AssertThrow(false, ExcNotImplemented());
                }
            }
        }

      if (as_dummy)
        status = NOX::StatusTest::Unconverged;

      return status;
    }

    NOX::StatusTest::StatusType
    getStatus() const override
    {
      return status;
    }

    virtual std::ostream &
    print(std::ostream &stream, int indent = 0) const override
    {
      (void)indent;

      std::string state_str;
      switch (state)
        {
          case NewtonSolverSolverControl::iterate:
            state_str = "iterate";
            break;
          case NewtonSolverSolverControl::failure:
            state_str = "failure";
            break;
          case NewtonSolverSolverControl::success:
            state_str = "success";
            break;
          default:
            AssertThrow(false, ExcNotImplemented());
        }

      for (int j = 0; j < indent; j++)
        stream << ' ';
      stream << status;
      stream << "check_iteration_status() = " << state_str
             << " (dummy = " << (as_dummy ? "yes" : "no") << ")";
      stream << std::endl;

      return stream;
    }

  private:
    std::function<NewtonSolverSolverControl::State(const unsigned int,
                                                   const double,
                                                   const VectorType &,
                                                   const VectorType &)>
      check_iteration_status = {};

    const bool as_dummy = false;

    NOX::StatusTest::StatusType      status;
    NewtonSolverSolverControl::State state;
  };

  template <typename VectorType>
  class NOXSolver : public NewtonSolver<VectorType>
  {
  public:
    NOXSolver(const NewtonSolverSolverControl &           statistics,
              const Teuchos::RCP<Teuchos::ParameterList> &non_linear_parameters,
              const NewtonSolverAdditionalData &          solver_data_in =
                NewtonSolverAdditionalData())
      : non_linear_parameters(non_linear_parameters)
      , solver_data(solver_data_in)
    {}

    void
    solve(VectorType &solution) const override
    {
      if (this->solver_data.reuse_preconditioner == false)
        clear();

      // create group
      const auto group =
        Teuchos::rcp(new internal::NOXWrapper::Group<VectorType>(
          solution,
          [&](const VectorType &src, VectorType &dst) {
            this->residual(src, dst);
          },
          [&](const VectorType &src, const bool /*flag*/) {
            const bool threshold_exceeded =
              (history_newton_iterations % solver_data.threshold_newton_iter ==
               0) ||
              (history_linear_iterations_last >
               solver_data.threshold_linear_iter);

            this->setup_jacobian(src,
                                 solver_data.do_update && threshold_exceeded);
          },
          [&](const VectorType &src, VectorType &dst) -> unsigned int {
            history_linear_iterations_last =
              this->solve_with_jacobian(src, dst);
            return 0; // dummy value
          }));

      // setup solver control
      const auto solver_control_norm_f_abs =
        Teuchos::rcp(new NOX::StatusTest::NormF(statistics.get_abs_tol()));

      const auto solver_control_norm_f_rel = Teuchos::rcp(
        new NOX::StatusTest::RelativeNormF(statistics.get_rel_tol()));

      const auto solver_control_max_iterations =
        Teuchos::rcp(new NOX::StatusTest::MaxIters(statistics.get_max_iter()));

      const auto info =
        Teuchos::rcp(new NOXCheck(this->check_iteration_status, true));

      auto check =
        Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

      check->addStatusTest(info);
      check->addStatusTest(solver_control_norm_f_abs);
      check->addStatusTest(solver_control_norm_f_rel);
      check->addStatusTest(solver_control_max_iterations);

      // create non-linear solver
      const auto solver =
        NOX::Solver::buildSolver(group, check, non_linear_parameters);

      // solve
      const auto status = solver->solve();

      AssertThrow(status == NOX::StatusTest::Converged,
                  ExcNewtonDidNotConverge("Newton"));

      history_newton_iterations = solver->getNumIterations();
    }

    void
    clear() const override
    {
      history_linear_iterations_last = 0;
      history_newton_iterations      = 0;
    }

  private:
    const Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters;
    const NewtonSolverAdditionalData           solver_data;

    mutable unsigned int history_linear_iterations_last = 0;
    mutable unsigned int history_newton_iterations      = 0;
  };
} // namespace NonLinearSolvers
