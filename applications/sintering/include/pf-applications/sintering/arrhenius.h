// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

#pragma once

#include <pf-applications/numerics/functions.h>

#include <memory>

namespace Sintering
{
  using namespace dealii;

  enum class ArrheniusUnit
  {
    Boltzmann,
    Gas
  };

  // Lightweight helper evaluating a generic Arrhenius relation
  //   value(T) = prefactor * exp(-activation_energy / (factor * T))
  // at a given time, where T is provided by a temperature function of time
  // and factor is the Boltzmann or the gas constant, depending on the units
  // used for the activation energy.
  class ArrheniusEvaluator
  {
  public:
    ArrheniusEvaluator(
      const double                        prefactor,
      const double                        activation_energy,
      std::shared_ptr<Function1D<double>> temperature,
      const ArrheniusUnit arrhenius_unit = ArrheniusUnit::Boltzmann)
      : prefactor(prefactor)
      , activation_energy(activation_energy)
      , temperature(temperature)
      , arrhenius_factor(arrhenius_unit == ArrheniusUnit::Boltzmann ? kb : R)
    {}

    double
    eval(const double time) const
    {
      const double T = temperature->value(time);

      return prefactor * std::exp(-activation_energy / (arrhenius_factor * T));
    }

  private:
    // Some constants
    static constexpr double kb = 8.617343e-5; // Boltzmann constant in eV/K
    static constexpr double R  = 8.314;       // Gas constant J / (mol K)

    const double                        prefactor;
    const double                        activation_energy;
    std::shared_ptr<Function1D<double>> temperature;
    const double                        arrhenius_factor;
  };

} // namespace Sintering
