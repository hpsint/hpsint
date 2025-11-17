// ---------------------------------------------------------------------
//
// Copyright (C) 2025 by the hpsint authors
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

#include <pf-applications/lac/solvers_linear_parameters.h>

#include <string>

namespace NonLinearSolvers
{
  struct NOXData
  {
    int         output_information             = 0;
    std::string direction_method               = "Newton";
    std::string line_search_method             = "Full Step";
    std::string line_search_interpolation_type = "Cubic";
    std::string nonlinear_solver               = "Line Search Based";
  };

  struct SNESData
  {
    std::string solver_name      = "newtonls";
    std::string line_search_name = "bt";
  };

  struct NonLinearData
  {
    int    nl_max_iter = 10;
    double nl_abs_tol  = 1.e-20;
    double nl_rel_tol  = 1.e-5;

    int          l_max_iter       = 1000;
    double       l_abs_tol        = 1.e-10;
    double       l_rel_tol        = 1.e-2;
    std::string  l_solver         = "GMRES";
    unsigned int l_bisgstab_tries = 30;

    bool         newton_do_update             = true;
    unsigned int newton_threshold_newton_iter = 100;
    unsigned int newton_threshold_linear_iter = 20;
    bool         newton_reuse_preconditioner  = true;
    bool         newton_use_damping           = true;

    std::string nonlinear_solver_type = "damped";

    bool fdm_jacobian_approximation       = false;
    bool fdm_precond_system_approximation = false;
    bool jacobi_free                      = false;

    unsigned int verbosity = 1;

    NOXData                  nox_data;
    SNESData                 snes_data;
    LinearSolvers::GMRESData gmres_data;
  };
} // namespace NonLinearSolvers
