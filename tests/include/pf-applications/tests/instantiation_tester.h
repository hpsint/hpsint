// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#define MAX_SINTERING_GRAINS 5

#include <pf-applications/sintering/instantiation.h>

#include <iostream>
#include <utility>
#include <vector>

namespace Test
{
  using Variants = std::vector<std::pair<unsigned int, bool>>;

  template <int c, int g>
  void
  dump_imp()
  {
    std::cout << "n_components = " << c << " | n_grains = " << g << std::endl;
  }

  template <bool extended, int n_ch>
  struct OperatorDummy
  {
    using T = OperatorDummy;

    OperatorDummy(unsigned int n_ac)
      : n_ac(n_ac)
    {}

    void
    dump() const
    {
      std::cout << "set_n_grains = " << n_ac << " -> ";
#define OPERATION(c, g) dump_imp<c, g>();
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    unsigned int
    n_components() const
    {
      return n_ac + n_ch;
    }

    void
    set_n_grains(unsigned int n_grains_new)
    {
      n_ac = n_grains_new;
    }

    // Dummy template to enable deduction context
    template <bool extended_ = extended>
    std::enable_if_t<extended_, unsigned int>
    n_grains() const
    {
      return n_ac;
    }

    // Dummy template to enable deduction context
    template <bool extended_ = extended>
    static constexpr std::enable_if_t<extended_, unsigned int>
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + n_ch;
    }

  private:
    unsigned int n_ac;
  };

  template <bool extended, int n_ch>
  void
  run_instantiation(const Variants &variants)
  {
    constexpr unsigned int        n_ac_init = 4;
    OperatorDummy<extended, n_ch> op(n_ac_init);

    auto test_variant = [&op](unsigned int n_ac, bool do_throw) {
      op.set_n_grains(n_ac);

      if (do_throw)
        try
          {
            op.dump();
          }
        catch (const ExcInvalidNumberOfComponents &ex)
          {
            ex.print_info(std::cout);
          }
      else
        op.dump();
    };

    std::cout << "Test operator with n_ch = " << n_ch
              << " and  n_ac = " << n_ac_init << std::endl;
    op.dump();

    for (const auto &variant : variants)
      test_variant(variant.first, variant.second);
  }
} // namespace Test