#ifndef SINTERING_PARTICLE_H_
#define SINTERING_PARTICLE_H_

#include <deal.II/base/point.h>

#include <vector>

#include <pf-applications/sintering/csv_reader.h>

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

  template <int dim>
  std::vector<Particle<dim>>
  read_particles(std::istream &stream)
  {
    std::vector<Particle<dim>> particles;

    unsigned int id_counter = 0;

    bool is_header_done = false;
    for (internal::CSVIterator loop(stream); loop != internal::CSVIterator(); ++loop)
      {
        if (!is_header_done)
          {
            is_header_done = true;
          }
        else
          {
            double x0 = std::stod(std::string((*loop)[0]));
            double y0 = std::stod(std::string((*loop)[1]));

            dealii::Point<dim> pt;
            if (dim == 2)
              {
                pt = dealii::Point<dim>(x0, y0);
              }
            else if (dim == 3)
              {
                double z0 = std::stod(std::string((*loop)[2]));

                pt = dealii::Point<dim>(x0, y0, z0);
              }

            double r0 = std::stod(std::string((*loop)[3]));

            Particle<dim> p{pt, r0, id_counter++, {}};
            particles.push_back(p);
          }
      }

    return particles;
  }

} // namespace Sintering

#endif