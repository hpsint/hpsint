#pragma once

namespace LinearSolvers
{
  struct GMRESData
  {
    std::string orthogonalization_strategy = "classical gram schmidt";
  };
} // namespace LinearSolvers
