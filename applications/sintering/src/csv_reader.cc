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

#include <pf-applications/sintering/csv_reader.h>

std::string_view
Sintering::internal::CSVRow::operator[](std::size_t index) const
{
  return std::string_view(&line[data[index] + 1],
                          data[index + 1] - (data[index] + 1));
}

std::size_t
Sintering::internal::CSVRow::size() const
{
  return data.size() - 1;
}

void
Sintering::internal::CSVRow::read_next_row(std::istream &str)
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

std::istream &
Sintering::internal::operator>>(std::istream &str, CSVRow &data)
{
  data.read_next_row(str);
  return str;
}

Sintering::internal::CSVIterator::CSVIterator(std::istream &str)
  : str(str.good() ? &str : nullptr)
{
  ++(*this);
}

Sintering::internal::CSVIterator::CSVIterator()
  : str()
{}

// Pre Increment
Sintering::internal::CSVIterator &
Sintering::internal::CSVIterator::operator++()
{
  if (str)
    {
      if (!((*str) >> row))
        {
          str = nullptr;
        }
    }
  return *this;
}

Sintering::internal::CSVIterator
Sintering::internal::CSVIterator::operator++(int)
{
  CSVIterator tmp(*this);
  ++(*this);
  return tmp;
}

const Sintering::internal::CSVRow &
Sintering::internal::CSVIterator::operator*() const
{
  return row;
}

const Sintering::internal::CSVRow *
Sintering::internal::CSVIterator::operator->() const
{
  return &row;
}

bool
Sintering::internal::CSVIterator::operator==(const CSVIterator &rhs)
{
  return ((this == &rhs) || ((this->str == nullptr) && (rhs.str == nullptr)));
}

bool
Sintering::internal::CSVIterator::operator!=(const CSVIterator &rhs)
{
  return !((*this) == rhs);
}
