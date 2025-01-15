#include <deal.II/base/conditional_ostream.h>

#include <pf-applications/tests/sintering_model.h>

#include <iostream>

namespace Test
{
  using namespace dealii;
  using namespace Sintering;

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename NonLinearOperator,
            typename FreeEnergy>
  void
  calc_residual(const bool enable_rbm)
  {
    using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

    SinteringModel<dim,
                   Number,
                   VectorType,
                   VectorizedArrayType,
                   NonLinearOperator,
                   FreeEnergy>
      sintering_model(enable_rbm);

    auto &nonlinear_operator = sintering_model.get_nonlinear_operator();
    auto &advection_operator = sintering_model.get_advection_operator();
    auto &dof_handler        = sintering_model.get_dof_handler();
    auto &solution           = sintering_model.get_solution();

    const auto comm = dof_handler.get_communicator();

    const bool is_zero_rank = Utilities::MPI::this_mpi_process(comm) == 0;
    ConditionalOStream pcout(std::cout, is_zero_rank);

    // Evaluate residual
    VectorType residual;
    nonlinear_operator.initialize_dof_vector(residual);

    if (enable_rbm)
      advection_operator.evaluate_forces(solution);
    nonlinear_operator.evaluate_nonlinear_residual(residual, solution);

    std::ostringstream ss;

    ss << "===== Output from rank " << Utilities::MPI::this_mpi_process(comm)
       << " (total = " << Utilities::MPI::n_mpi_processes(comm)
       << ") =====" << std::endl;

    for (unsigned int c = 0; c < residual.n_blocks(); ++c)
      {
        ss << "block " << c << ": ";
        residual.block(c).print(ss);
      }

    auto all_prints = Utilities::MPI::gather(comm, ss.str());

    for (const auto &entry : all_prints)
      pcout << entry;
  }
} // namespace Test
