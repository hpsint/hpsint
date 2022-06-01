namespace dealii::parallel
{
  template <int dim, int spacedim = dim>
  class Helper
  {
  public:
    Helper(Triangulation<dim, spacedim> &triangulation)
    {
      reinit(triangulation);

      const auto fu = [&]() { this->reinit(triangulation); };

      triangulation.signals.post_p4est_refinement.connect(fu);
      triangulation.signals.post_distributed_refinement.connect(fu);
      triangulation.signals.pre_distributed_repartition.connect(fu);
      triangulation.signals.post_distributed_repartition.connect(fu);
    }

    void
    reinit(const Triangulation<dim, spacedim> &triangulation)
    {
      if (dim == 3)
        {
          this->line_to_cells.clear();

          const unsigned int n_raw_lines = triangulation.n_raw_lines();
          this->line_to_cells.resize(n_raw_lines);

          // In 3D, we can have DoFs on only an edge being constrained (e.g. in
          // a cartesian 2x2x2 grid, where only the upper left 2 cells are
          // refined). This sets up a helper data structure in the form of a
          // mapping from edges (i.e. lines) to neighboring cells.

          // Mapping from an edge to which children that share that edge.
          const unsigned int line_to_children[12][2] = {{0, 2},
                                                        {1, 3},
                                                        {0, 1},
                                                        {2, 3},
                                                        {4, 6},
                                                        {5, 7},
                                                        {4, 5},
                                                        {6, 7},
                                                        {0, 4},
                                                        {1, 5},
                                                        {2, 6},
                                                        {3, 7}};

          std::vector<std::vector<
            std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                      unsigned int>>>
            line_to_inactive_cells(n_raw_lines);

          // First add active and inactive cells to their lines:
          for (const auto &cell : triangulation.cell_iterators())
            {
              for (unsigned int line = 0;
                   line < GeometryInfo<3>::lines_per_cell;
                   ++line)
                {
                  const unsigned int line_idx = cell->line(line)->index();
                  if (cell->is_active())
                    line_to_cells[line_idx].push_back(
                      std::make_pair(cell, line));
                  else
                    line_to_inactive_cells[line_idx].push_back(
                      std::make_pair(cell, line));
                }
            }

          // Now, we can access edge-neighboring active cells on same level to
          // also access of an edge to the edges "children". These are found
          // from looking at the corresponding edge of children of inactive edge
          // neighbors.
          for (unsigned int line_idx = 0; line_idx < n_raw_lines; ++line_idx)
            {
              if ((line_to_cells[line_idx].size() > 0) &&
                  line_to_inactive_cells[line_idx].size() > 0)
                {
                  // We now have cells to add (active ones) and edges to which
                  // they should be added (inactive cells).
                  const auto &inactive_cell =
                    line_to_inactive_cells[line_idx][0].first;
                  const unsigned int neighbor_line =
                    line_to_inactive_cells[line_idx][0].second;

                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      const auto &child = inactive_cell->child(
                        line_to_children[neighbor_line][c]);
                      const unsigned int child_line_idx =
                        child->line(neighbor_line)->index();

                      // Now add all active cells
                      for (const auto &cl : line_to_cells[line_idx])
                        line_to_cells[child_line_idx].push_back(cl);
                    }
                }
            }
        }
    }


    bool
    is_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      return is_face_constrained(cell) || is_edge_constrained(cell);
    }

    bool
    is_face_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      if (cell->is_locally_owned())
        for (unsigned int f : cell->face_indices())
          if (!cell->at_boundary(f) &&
              (cell->level() > cell->neighbor(f)->level()))
            return true;

      return false;
    }

    bool
    is_edge_constrained(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
      const
    {
      if (dim == 3)
        if (cell->is_locally_owned())
          for (const auto line : cell->line_indices())
            for (const auto &other_cell :
                 line_to_cells[cell->line(line)->index()])
              if (cell->level() > other_cell.first->level())
                return true;

      return false;
    }

  private:
    std::vector<std::vector<
      std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                unsigned int>>>
      line_to_cells;
  };

  template <int dim, int spacedim = dim>
  std::function<
    unsigned int(const typename Triangulation<dim, spacedim>::cell_iterator &,
                 const typename Triangulation<dim, spacedim>::CellStatus)>
  hanging_nodes_weighting(const Helper<dim, spacedim> &helper,
                          const double                 weight)
  {
    return [&helper, weight](const auto &cell, const auto &) -> unsigned int {
      if (cell->is_active() == false || cell->is_locally_owned() == false)
        return 10000;

      if (helper.is_constrained(cell))
        return 10000 * weight;
      else
        return 10000;
    };
  }


} // namespace dealii::parallel
