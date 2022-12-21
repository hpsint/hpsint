#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <pf-applications/lac/dynamic_block_vector.h>

using namespace dealii;


template <typename VectorType>
constexpr bool is_dealii_compatible_vector =
  std::is_same<
    VectorType,
    LinearAlgebra::distributed::Vector<typename VectorType::value_type,
                                       MemorySpace::Host>>::value ||
  std::is_same<VectorType,
               LinearAlgebra::distributed::BlockVector<
                 typename VectorType::value_type>>::value ||
  std::is_same<VectorType,
               LinearAlgebra::distributed::DynamicBlockVector<
                 typename VectorType::value_type>>::value;

template <typename VectorType,
          std::enable_if_t<!is_dealii_compatible_vector<VectorType>, VectorType>
            * = nullptr>
void
test(const VectorType &)
{
  std::cout << "Fail!" << std::endl;
}

template <typename VectorType,
          std::enable_if_t<is_dealii_compatible_vector<VectorType>, VectorType>
            * = nullptr>
void
test(const VectorType &)
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
