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

#pragma once

#include <boost/algorithm/string/join.hpp>

class ScopedName
{
public:
  ScopedName(const std::string name)
    : name(name)
  {
    path.push_back(name);
  }

  ~ScopedName()
  {
    AssertThrow(path.back() == name, dealii::ExcInternalError());
    path.pop_back();
  }

  operator std::string() const
  {
    return boost::algorithm::join(path, "::");
    ;
  }

private:
  const std::string                      name;
  inline static std::vector<std::string> path;
};
