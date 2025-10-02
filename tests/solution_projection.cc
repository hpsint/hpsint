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

#define MAX_SINTERING_GRAINS 2

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/numerics/output.h>

#include <pf-applications/sintering/projection.h>

using namespace dealii;
using namespace Sintering;

using Number     = double;
using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution()
    : Function<dim>(1)
    , factor(0)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const final
  {
    (void)component;
    return factor * (p[0] * p[1] * (p[2] + 2));
  }

  void
  set_factor(const double f)
  {
    factor = f;
  }

private:
  double factor;
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const MPI_Comm comm = MPI_COMM_WORLD;

  const bool         is_zero_rank = Utilities::MPI::this_mpi_process(comm) == 0;
  ConditionalOStream pcout(std::cout, is_zero_rank);

  const unsigned int dim = 3;

  FE_Q<dim>      fe{1};
  MappingQ1<dim> mapping;

  Point<dim> bottom_left{1, 1, -1};
  Point<dim> top_right{2, 2, 1};

  std::vector<unsigned int> subdivisions{2, 2, 4};

  parallel::distributed::Triangulation<dim> tria(comm);
  dealii::GridGenerator::subdivided_hyper_rectangle(
    tria, subdivisions, bottom_left, top_right, true);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Init solution vector
  VectorType solution(2);

  Solution<dim> initial_solution;

  const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler),
    dof_handler.get_mpi_communicator());

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    solution.block(c).reinit(partitioner);

  solution.zero_out_ghost_values();

  for (unsigned int c = 0; c < solution.n_blocks(); ++c)
    {
      initial_solution.set_factor(c + 1);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               initial_solution,
                               solution.block(c));
    }

  const unsigned int direction = 2;
  const double       location  = 0;

  const auto projection = Postprocessors::build_projection(dof_handler,
                                                           solution,
                                                           direction,
                                                           location);

  std::ostringstream ss;

  ss << "===== Projected solution from rank "
     << Utilities::MPI::this_mpi_process(comm) << " =====" << std::endl;

  for (unsigned int c = 0; c < projection->solution.size(); ++c)
    {
      ss << "block " << c << ": ";
      projection->solution[c].print(ss);
    }

  auto all_prints_solution = Utilities::MPI::gather(comm, ss.str());

  for (const auto &entry : all_prints_solution)
    pcout << entry;

  ss.str(std::string());
  ss << "===== Projected mesh from rank "
     << Utilities::MPI::this_mpi_process(comm) << " =====" << std::endl;

  for (const auto &cell : projection->dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        ss << "cell id = " << cell->active_cell_index() << ":" << std::endl;
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          {
            ss << " vertex id = " << cell->vertex_index(v);
            ss << ", coords = " << cell->vertex(v) << std::endl;
          }
      }

  auto all_prints_mesh = Utilities::MPI::gather(comm, ss.str());

  for (const auto &entry : all_prints_mesh)
    pcout << entry;
}