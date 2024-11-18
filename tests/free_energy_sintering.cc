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

#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>

#include <pf-applications/sintering/free_energy.h>

#include <array>
#include <vector>

using namespace hpsint;
using namespace dealii;
using namespace Sintering;

int
main()
{
  FreeEnergy<double> free_energy(2, 3);

  const double              c  = 0.95;
  const double              mu = 0.0;
  const std::vector<double> etas({0.3, 0.15, 0.47, 0.});

  constexpr auto n_grains = 4;

  auto test_evaluator = [&](const auto &evaluator) {
    std::cout << "f = " << evaluator.f() << std::endl;
    std::cout << "df_dc = " << evaluator.df_dc() << std::endl;

    for (unsigned int i = 0; i < n_grains; ++i)
      std::cout << "df_detai_" << i << " = " << evaluator.df_detai(etas[i])
                << std::endl;

    std::cout << "d2f_dc2 = " << evaluator.d2f_dc2() << std::endl;

    for (unsigned int i = 0; i < n_grains; ++i)
      std::cout << "d2f_dcdetai_" << i << " = "
                << evaluator.d2f_dcdetai(etas[i]) << std::endl;

    for (unsigned int i = 0; i < n_grains; ++i)
      std::cout << "d2f_detai2_" << i << " = " << evaluator.d2f_detai2(etas[i])
                << std::endl;

    for (unsigned int i = 0; i < n_grains; ++i)
      for (unsigned int j = i + 1; j < n_grains; ++j)
        std::cout << "d2f_detaidetaj_" << i << "_" << j << " = "
                  << evaluator.d2f_detaidetaj(etas[i], etas[j]) << std::endl;
  };

  // State as std::vector
  std::vector<double> state_vec{c, mu};
  std::copy(etas.begin(), etas.end(), std::back_inserter(state_vec));

  const auto free_energy_eval_vec =
    free_energy.template eval<EnergyZero>(state_vec, n_grains);

  std::cout << "state via std::vector:" << std::endl;
  test_evaluator(free_energy_eval_vec);
  std::cout << std::endl;

  // State as std::array with n_components auto-derived
  std::array<double, n_grains + 2> state_arr{{c, mu}};
  std::copy(etas.begin(), etas.end(), state_arr.begin() + 2);

  const auto free_energy_eval_arr_auto =
    free_energy.template eval<EnergyZero>(state_arr);
  std::cout << "state via std::array with n_components auto-derived:"
            << std::endl;
  test_evaluator(free_energy_eval_arr_auto);
  std::cout << std::endl;

  // State as std::array with n_grains specified
  const auto free_energy_eval_arr_manual =
    free_energy.template eval<EnergyZero, n_grains>(state_arr);
  std::cout << "state via std::array with n_grains specified:" << std::endl;
  test_evaluator(free_energy_eval_arr_manual);
  std::cout << std::endl;

  // State as raw pointer with n_grains via template param
  const auto *ptr = state_vec.data();
  const auto  free_energy_eval_ptr_tpl =
    free_energy.template eval<EnergyZero, n_grains>(ptr);
  std::cout << "state raw pointer with n_grains via template param:"
            << std::endl;
  test_evaluator(free_energy_eval_ptr_tpl);
  std::cout << std::endl;

  // State as raw pointer with n_grains via ctor param
  const auto  free_energy_eval_ptr_par =
    free_energy.template eval<EnergyZero>(ptr, n_grains);
  std::cout << "state raw pointer with n_grains via ctor param:" << std::endl;
  test_evaluator(free_energy_eval_ptr_par);
}