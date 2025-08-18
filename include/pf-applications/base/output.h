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

#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <ostream>
#include <sstream>

namespace hpsint
{
  using namespace dealii;

  namespace internal
  {
    template <typename Stream>
    struct Dumper
    {
      template <typename T>
      static void
      print(const T &object, std::ostream &stream)
      {
        object.print(stream);
      }
    };

    template <>
    struct Dumper<ConditionalOStream>
    {
      template <typename T>
      static void
      print(const T &object, ConditionalOStream &stream)
      {
        std::stringstream ss;
        object.print(ss);
        stream << ss.str();
      }
    };
  } // namespace internal

  template <typename T, typename Stream>
  void
  print(const T &object, Stream &stream)
  {
    internal::Dumper<Stream>::print(object, stream);
  }

  std::string
  concatenate_strings(const int argc, char **argv);

} // namespace hpsint