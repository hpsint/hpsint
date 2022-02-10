#ifndef SINTERING_PARTICLE_H_
#define SINTERING_PARTICLE_H_

#include <deal.II/base/point.h>

#include <vector>

namespace Sintering
{
  template <int dim>
  struct Particle
  {
    dealii::Point<dim>        center;
    double                    radius;
    unsigned int              id;
    std::vector<unsigned int> neighbours;
  };
} // namespace Sintering

#endif