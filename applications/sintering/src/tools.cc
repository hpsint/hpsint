// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <pf-applications/sintering/tools.h>

using namespace dealii;

std::vector<unsigned int>
Sintering::get_primes(unsigned int start, unsigned int end)
{
  std::vector<unsigned int> primes;

  for (unsigned int i = start; i <= end; ++i)
    {
      // Skip 0 and 1
      if (i == 1 || i == 0)
        continue;

      bool is_prime = true;

      // Iterate to check if i is prime
      for (unsigned int j = 2; j <= i / 2; ++j)
        if (i % j == 0)
          {
            is_prime = false;
            break;
          }

      if (is_prime)
        primes.push_back(i);
    }
  return primes;
}

std::pair<unsigned int, unsigned int>
Sintering::decompose_to_prime_tuple(const unsigned int n_ref,
                                    const unsigned     max_prime)
{
  const auto primes = get_primes(2, max_prime);

  unsigned int optimal_prime      = 0;
  unsigned int n_refinements      = 0;
  unsigned int min_elements_delta = numbers::invalid_unsigned_int;
  for (const auto &p : primes)
    {
      const unsigned int s =
        static_cast<unsigned int>(std::ceil(std::log2(n_ref / p)));
      const unsigned int n_current     = p * std::pow(2, s);
      const unsigned int current_delta = static_cast<unsigned int>(
        std::abs(static_cast<int>(n_current - n_ref)));

      if (current_delta < min_elements_delta)
        {
          min_elements_delta = current_delta;
          optimal_prime      = p;
          n_refinements      = s;
        }
    }

  return std::make_pair(optimal_prime, n_refinements);
}

unsigned int
Sintering::internal::adjust_divisions_to_primes(
  const unsigned int         max_prime,
  std::vector<unsigned int> &subdivisions)
{
  // Further reduce the number of initial subdivisions
  unsigned int n_refinements_base = 0;
  if (max_prime > 0)
    {
      std::vector<unsigned int> refinements(subdivisions.size());

      for (unsigned int d = 0; d < subdivisions.size(); ++d)
        {
          const auto pair =
            decompose_to_prime_tuple(subdivisions[d], max_prime);

          subdivisions[d] = pair.first;
          refinements[d]  = pair.second;
        }

      n_refinements_base =
        *std::min_element(refinements.begin(), refinements.end());

      for (unsigned int d = 0; d < subdivisions.size(); ++d)
        subdivisions[d] *= std::pow(2, refinements[d] - n_refinements_base);
    }

  return n_refinements_base;
}

Sintering::EnergyCoefficients
Sintering::compute_energy_params(const double surface_energy,
                                 const double gb_energy,
                                 const double interface_width,
                                 const double length_scale,
                                 const double energy_scale)
{
  const double scaled_gb_energy =
    gb_energy / energy_scale * length_scale * length_scale;

  const double scaled_surface_energy =
    surface_energy / energy_scale * length_scale * length_scale;

  const double kappa_c = 3.0 / 4.0 *
                         (2.0 * scaled_surface_energy - scaled_gb_energy) *
                         interface_width;
  const double kappa_p = 3.0 / 4.0 * scaled_gb_energy * interface_width;

  const double A =
    (12.0 * scaled_surface_energy - 7.0 * scaled_gb_energy) / interface_width;
  const double B = scaled_gb_energy / interface_width;

  EnergyCoefficients params{A, B, kappa_c, kappa_p};

  return params;
}

std::string
Sintering::concatenate_strings(const int argc, char **argv)
{
  std::string result = std::string(argv[0]);

  for (int i = 1; i < argc; ++i)
    result = result + " " + std::string(argv[i]);

  return result;
}
