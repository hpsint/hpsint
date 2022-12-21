#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>

using namespace dealii;

template <typename VectorType>
void
test(const VectorType &)
{
  std::cout << "Fail!" << std::endl;
}

template <typename Number>
void
test(const LinearAlgebra::distributed::Vector<Number> &)
{
  std::cout << "Success!" << std::endl;
}

template <typename Number>
void
test(const LinearAlgebra::distributed::BlockVector<Number> &)
{
  std::cout << "Success!" << std::endl;
}

template <typename Number>
void
test(const LinearAlgebra::distributed::DynamicBlockVector<Number> &)
{
  std::cout << "Success!" << std::endl;
}

int
main()
{
  {
    LinearAlgebra::distributed::Vector<double> vec;
    test(vec);
  }

  {
    LinearAlgebra::distributed::BlockVector<double> vec;
    test(vec);
  }

  {
    LinearAlgebra::distributed::DynamicBlockVector<double> vec;
    test(vec);
  }

  {
    Vector<double> vec;
    test(vec);
  }
}
