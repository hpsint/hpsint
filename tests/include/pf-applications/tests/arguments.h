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


#include <iostream>
#include <utility>
#include <vector>

namespace Test
{
  class MyArguments
  {
  public:
    MyArguments()
    {
      my_argv.push_back(nullptr);
    }

    template <typename Iterator>
    MyArguments(Iterator begin, Iterator end)
      : arguments(begin, end)
    {
      std::transform(arguments.begin(),
                     arguments.end(),
                     std::back_inserter(my_argv),
                     [](std::string &arg) {
                       return static_cast<char *>(arg.data());
                     });
      my_argv.push_back(nullptr);
    }

    MyArguments(const MyArguments &other) = delete;

    MyClass &
    operator=(const MyClass &) = delete;

    char **
    argv()
    {
      return my_argv.data();
    }

    int
    argc() const
    {
      return my_argv.size() ? my_argv.size() - 1 : 0;
    }

  private:
    std::vector<char *>      my_argv;
    std::vector<std::string> arguments;
  };
} // namespace Test
