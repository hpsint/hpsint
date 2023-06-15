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
      operator[](std::size_t index) const
      {
        return std::string_view(&line[data[index] + 1],
                                data[index + 1] - (data[index] + 1));
      }
      std::size_t
      size() const
      {
        return data.size() - 1;
      }
      void
      read_next_row(std::istream &str)
      {
        std::getline(str, line);

        data.clear();
        data.emplace_back(-1);
        std::string::size_type pos = 0;
        while ((pos = line.find(',', pos)) != std::string::npos)
          {
            data.emplace_back(pos);
            ++pos;
          }
        // This checks for a trailing comma with no data after it.
        pos = line.size();
        data.emplace_back(pos);
      }

    private:
      std::string      line;
      std::vector<int> data;
    };

    std::istream &
    operator>>(std::istream &str, CSVRow &data)
    {
      data.read_next_row(str);
      return str;
    }

    class CSVIterator
    {
    public:
      using iterator_category = std::input_iterator_tag;
      using value_type        = CSVRow;
      using difference_type   = std::size_t;
      using pointer           = CSVRow *;
      using reference         = CSVRow &;

      CSVIterator(std::istream &str)
        : str(str.good() ? &str : NULL)
      {
        ++(*this);
      }
      CSVIterator()
        : str(NULL)
      {}

      // Pre Increment
      CSVIterator &
      operator++()
      {
        if (str)
          {
            if (!((*str) >> row))
              {
                str = NULL;
              }
          }
        return *this;
      }
      // Post increment
      CSVIterator
      operator++(int)
      {
        CSVIterator tmp(*this);
        ++(*this);
        return tmp;
      }
      CSVRow const &
      operator*() const
      {
        return row;
      }
      CSVRow const *
      operator->() const
      {
        return &row;
      }

      bool
      operator==(CSVIterator const &rhs)
      {
        return ((this == &rhs) || ((this->str == NULL) && (rhs.str == NULL)));
      }
      bool
      operator!=(CSVIterator const &rhs)
      {
        return !((*this) == rhs);
      }

    private:
      std::istream *str;
      CSVRow        row;
    };
  } // namespace internal
} // namespace Sintering
