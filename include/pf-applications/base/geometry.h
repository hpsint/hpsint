
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/point.h>

namespace dealii
{
  template <int dim>
  BoundingBox<dim>
  create_bounding_box_around_point(const Point<dim> &center,
                                   const double      radius)
  {
    Point<dim> lower_left = center;
    Point<dim> top_right  = center;
    for (unsigned int d = 0; d < dim; ++d)
      {
        lower_left[d] -= radius;
        top_right[d] += radius;
      }

    return BoundingBox<dim>({lower_left, top_right});
  }
} // namespace dealii