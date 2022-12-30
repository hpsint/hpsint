#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_snes.h>
#include <deal.II/lac/petsc_vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>

using namespace dealii;

template <typename VectorType>
struct PETSCVectorTraits;

template <typename Number>
struct PETSCVectorTraits<LinearAlgebra::distributed::Vector<Number>>
{
  using VectorType  = LinearAlgebra::distributed::Vector<Number>;
  using PVectorType = PETScWrappers::MPI::Vector;
  using PMatrixType = PETScWrappers::MatrixBase; // not needed!?

  static void
  copy(VectorType &dst, const PVectorType &src)
  {
    for (const auto i : dst.locally_owned_elements())
      dst[i] = src[i];
  }

  static void
  copy(PVectorType &dst, const VectorType &src)
  {
    for (const auto i : src.locally_owned_elements())
      dst[i] = src[i];
    dst.compress(VectorOperation::insert);
  }

  static std::shared_ptr<PVectorType>
  create(const VectorType &vector)
  {
    return std::make_shared<PVectorType>(vector.locally_owned_elements(),
                                         vector.get_mpi_communicator());
  }
};

template <typename Number>
struct PETSCVectorTraits<LinearAlgebra::distributed::DynamicBlockVector<Number>>
{
  using VectorType  = LinearAlgebra::distributed::DynamicBlockVector<Number>;
  using PVectorType = PETScWrappers::MPI::BlockVector;
  using PMatrixType = PETScWrappers::MPI::BlockSparseMatrix; // not needed!?

  static void
  copy(VectorType &dst, const PVectorType &src)
  {
    for (unsigned int b = 0; b < src.n_blocks(); ++b)
      for (const auto i : dst.block(b).locally_owned_elements())
        dst.block(b)[i] = src.block(b)[i];
  }

  static void
  copy(PVectorType &dst, const VectorType &src)
  {
    for (unsigned int b = 0; b < src.n_blocks(); ++b)
      for (const auto i : src.block(b).locally_owned_elements())
        dst.block(b)[i] = src.block(b)[i];
    dst.compress(VectorOperation::insert);
  }

  static std::shared_ptr<PVectorType>
  create(const VectorType &vector)
  {
    std::vector<IndexSet> index_sets(vector.n_blocks());

    for (unsigned int b = 0; b < vector.n_blocks(); ++b)
      index_sets[b] = vector.block(b).locally_owned_elements();

    return std::make_shared<PVectorType>(
      index_sets, vector.block(0).get_mpi_communicator());
  }
};

template <typename VectorType>
class MyNonlinearSolver
{
private:
  // TODO: generalize for block vectors
  using VectorTraits = PETSCVectorTraits<VectorType>;
  using PVectorType  = typename VectorTraits::PVectorType;
  using PMatrixType  = typename VectorTraits::PMatrixType; // not needed!?

public:
  void
  solve(VectorType &vector)
  {
    PETScWrappers::NonlinearSolver<PVectorType, PMatrixType> solver;

    // create a temporal PETSc vector
    auto pvector = VectorTraits::create(vector);

    // create temporal deal.II vectors
    VectorType tmp_0, tmp_1;
    tmp_0.reinit(vector);
    tmp_1.reinit(vector);

    // wrap deal.II functions
    solver.residual = [&](const PVectorType &X, PVectorType &F) -> int {
      VectorTraits::copy(tmp_0, X);
      const auto ierr = this->residual(tmp_0, tmp_1);
      VectorTraits::copy(F, tmp_1);
      return ierr;
    };

    solver.setup_jacobian = [&](const PVectorType &X) -> int {
      VectorTraits::copy(tmp_1, X);
      const auto ierr = this->setup_jacobian(tmp_1);
      return ierr;
    };

    solver.solve_for_jacobian_system = [&](const PVectorType &src,
                                           PVectorType &      dst) -> int {
      VectorTraits::copy(tmp_0, src);
      const auto ierr = this->solve_for_jacobian_system(tmp_0, tmp_1);
      VectorTraits::copy(dst, tmp_1);
      return ierr;
    };

    // solve
    VectorTraits::copy(*pvector, vector);
    solver.solve(*pvector);
    VectorTraits::copy(vector, *pvector);
  }

  std::function<int(const VectorType &X, VectorType &F)> residual;
  std::function<int(const VectorType &X)>                setup_jacobian;
  std::function<int(const VectorType &src, VectorType &dst)>
    solve_for_jacobian_system;
};

template <typename VectorType>
void
test()
{
  MyNonlinearSolver<VectorType> solver;

  solver.residual = [&](const VectorType &X, VectorType &F) -> int {
    (void)X;
    (void)F;

    return 0;
  };

  solver.setup_jacobian = [&](const VectorType &X) -> int {
    (void)X;

    return 0;
  };

  solver.solve_for_jacobian_system = [&](const VectorType &src,
                                         VectorType &      dst) -> int {
    (void)src;
    (void)dst;

    return 0;
  };

  VectorType solution;
  solver.solve(solution);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (false)
    {
      test<LinearAlgebra::distributed::Vector<double>>();
      test<LinearAlgebra::distributed::DynamicBlockVector<double>>();
    }
}
