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

#include <deal.II/base/mg_level_object.h>

/**
 * Similar to the dealii::MemoryConsumption namespace but values
 * of smared pointers are counted.
 */
namespace MyMemoryConsumption
{
  template <typename T>
  inline std::enable_if_t<std::is_fundamental<T>::value, std::size_t>
  memory_consumption(const T &t);

  template <typename T>
  inline std::enable_if_t<!(std::is_fundamental<T>::value ||
                            std::is_pointer<T>::value),
                          std::size_t>
  memory_consumption(const T &t);

  template <typename T>
  inline std::size_t
  memory_consumption(const std::shared_ptr<T> &ptr);

  template <typename T>
  inline std::size_t
  memory_consumption(const std::unique_ptr<T> &ptr);

  template <typename T>
  inline std::size_t
  memory_consumption(const std::vector<T> &vector);

  template <typename T>
  inline std::size_t
  memory_consumption(const dealii::MGLevelObject<T> &mg_level_object);



  template <typename T>
  inline std::enable_if_t<std::is_fundamental<T>::value, std::size_t>
  memory_consumption(const T &)
  {
    return sizeof(T);
  }

  template <typename T>
  inline std::size_t
  memory_consumption(const std::shared_ptr<T> &ptr)
  {
    if (ptr)
      return memory_consumption(*ptr);
    else
      return 0.0;
  }

  template <typename T>
  inline std::size_t
  memory_consumption(const std::unique_ptr<T> &ptr)
  {
    if (ptr)
      return memory_consumption(*ptr);
    else
      return 0.0;
  }

  template <typename T>
  inline std::size_t
  memory_consumption(const std::vector<T> &vector)
  {
    std::size_t size = 0;

    for (const auto &entry : vector)
      size += memory_consumption(entry);

    return size;
  }

  template <typename T>
  inline std::size_t
  memory_consumption(const dealii::MGLevelObject<T> &mg_level_object)
  {
    std::size_t size = 0;

    for (auto l = mg_level_object.min_level(); l <= mg_level_object.max_level();
         ++l)
      size += memory_consumption(mg_level_object[l]);

    return size;
  }

  template <typename T>
  inline std::enable_if_t<!(std::is_fundamental<T>::value ||
                            std::is_pointer<T>::value),
                          std::size_t>
  memory_consumption(const T &t)
  {
    return t.memory_consumption();
  }

} // namespace MyMemoryConsumption