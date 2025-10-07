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

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace Sintering
{
  namespace internal
  {
    class CSVRow
    {
    public:
      std::string_view
      operator[](std::size_t index) const;

      std::size_t
      size() const;

      void
      read_next_row(std::istream &str);

    private:
      std::string      line;
      std::vector<int> data;
    };

    std::istream &
    operator>>(std::istream &str, CSVRow &data);

    class CSVIterator
    {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type        = CSVRow;
      using difference_type   = std::size_t;
      using pointer           = CSVRow *;
      using reference         = CSVRow &;

      CSVIterator(std::istream &str);

      CSVIterator();

      // Pre Increment
      CSVIterator &
      operator++();

      // Post increment
      CSVIterator
      operator++(int);

      const CSVRow &
      operator*() const;

      const CSVRow *
      operator->() const;

      bool
      operator==(const CSVIterator &rhs);

      bool
      operator!=(const CSVIterator &rhs);

    private:
      std::istream *str;
      CSVRow        row;
    };
  } // namespace internal
} // namespace Sintering
