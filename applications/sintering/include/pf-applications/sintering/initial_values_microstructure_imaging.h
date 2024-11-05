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

#include <deal.II/base/bounding_box.h>

#include <pf-applications/sintering/csv_reader.h>
#include <pf-applications/sintering/initial_values.h>
#include <pf-applications/sintering/initial_values_microstructure.h>

namespace Sintering
{
  using namespace dealii;

  /* This class represents custom microstructure read from a file. Works only
   * for the 2D case. The file has the following format:
   *
   * N                   - number of grains
   * op_0 op_1 ... op_N  - order parameters indices for each grain
   * x_0_0 y_0_0 x_0_1 y_0_1 ... x_0_M0 y_0_M0 - coordinates of the contour
   * M0 points of grain 0
   * ....
   * x_N_0 y_N_0 x_N_1 y_N_1 ... x_N_MN y_N_MN - coordinates of the contour
   * MN points of grain N
   *
   * The points of each contour have to be written in the clockwise or
   * counterclockwise order, the first point is not not repeated at the end.
   */
  class InitialValuesMicrostructureImaging : public InitialValuesMicrostructure
  {
  public:
    InitialValuesMicrostructureImaging(std::istream &     stream,
                                       const double       interface_width = 0.,
                                       const unsigned int op_offset       = 2)
      : InitialValuesMicrostructure(interface_width, op_offset)
    {
      unsigned int row_counter = 0;
      unsigned int n_grains    = 0;

      for (internal::CSVIterator loop(stream); loop != internal::CSVIterator();
           ++loop, ++row_counter)
        {
          if (row_counter == 0)
            {
              n_grains = std::stoi(std::string((*loop)[0]));
              grains.reserve(n_grains);
            }
          else if (row_counter == 1)
            {
              AssertDimension(n_grains, loop->size());

              for (unsigned int i = 0; i < loop->size(); ++i)
                {
                  const auto color = std::stoi(std::string((*loop)[i]));
                  grains.emplace_back(color, interface_width);
                  order_parameter_to_grains[color].push_back(i);
                }
            }
          else
            {
              MicroSegment segment;

              for (unsigned int i = 0; i < loop->size(); i += 2)
                {
                  segment.vertices.emplace_back(
                    std::stod(std::string((*loop)[i])),
                    std::stod(std::string((*loop)[i + 1])));
                }

              segment.vertices.push_back(segment.vertices[0]);
              segment.box = BoundingBox<2>(segment.vertices);
              segment.box.extend(interface_width);

              grains[row_counter - 2].add_segment(std::move(segment));
            }
        }
    }
  };
} // namespace Sintering
