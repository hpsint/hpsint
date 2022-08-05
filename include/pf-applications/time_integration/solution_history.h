#pragma once

#include <pf-applications/lac/dynamic_block_vector.h>

#include <functional>

namespace TimeIntegration
{
  using namespace dealii;

  template <typename VectorType>
  class SolutionHistory
  {
    template <typename T>
    using n_blocks_t = decltype(std::declval<T const>().n_blocks());

  public:
    SolutionHistory(unsigned int size)
      : solutions(size)
    {
      for (unsigned int i = 0; i < solutions.size(); ++i)
        solutions[i] = std::make_shared<VectorType>();
    }

    void initialize_dof_vectors(std::function<void(VectorType &)> f)
    {
      for (unsigned int i = 0; i < solutions.size(); ++i)
          f(*solutions[i]);
    }

    template <typename T = VectorType, std::enable_if_t<dealii::internal::is_supported_operation<n_blocks_t, T>, bool>>
    unsigned int n_blocks_total()
    {
      unsigned int n_blocks = 0;
      for (const auto &sol : solutions)
        n_blocks += sol->n_blocks();

      return n_blocks;
    }

    template <typename T = VectorType, std::enable_if_t<!dealii::internal::is_supported_operation<n_blocks_t, T>, bool>>
    unsigned int n_blocks_total()
    {
      return 0;
    }

    void
    commit_old_solutions() const
    {
      for (int i = solutions.size() - 2; i >= 1; --i) // 1 or 0 ???
        {
          solutions[i]->zero_out_ghost_values();
          *solutions[i + 1] = *solutions[i];
          solutions[i + 1]->update_ghost_values();
        }
    }

    void
    set_recent_old_solution(const VectorType &src) const
    {
      Assert(src.has_ghost_elements() == false, ExcInternalError());

      *solutions[1] = src;
      solutions[1]->update_ghost_values();
    }

    const VectorType &
    get_recent_old_solution() const
    {
      solutions[1]->zero_out_ghost_values();
      return *solutions[1];
    }

    const VectorType &
    get_current_solution() const
    {
      solutions[0]->zero_out_ghost_values();
      return *solutions[0];
    }

    VectorType &
    get_current_solution()
    {
      solutions[0]->zero_out_ghost_values();
      return *solutions[0];
    }

    std::vector<std::shared_ptr<VectorType>>
    get_old_solutions(bool skip_first = true) const
    {
      const unsigned int offset = skip_first ? 2 : 1;

      return std::vector<std::shared_ptr<VectorType>>(solutions.begin() +
                                                        offset,
                                                      solutions.end());
    }

  private:
    std::vector<std::shared_ptr<VectorType>> solutions;
  };
} // namespace TimeIntegration