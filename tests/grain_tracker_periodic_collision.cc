// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the hpsint authors
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

/* Behavior of the grain tracker in a periodic domain if grains sharing an
 * order parameter come close to each other across the periodic boundary:
 *
 * 1. The grains are close enough to be recognized as colliding. They are
 *    moved to different order parameters and do not merge later on.
 *
 * 2. The grains were not recognized as colliding and their fields merge. The
 *    tracker has to report that.
 *
 * 3. A single grain crosses the periodic boundary. It becomes a grain having
 *    two segments and keeps its id. This must not be reported.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <pf-applications/tests/periodic_grains_tester.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace dealii;
using namespace PeriodicGrainsTester;

using GrainTracker::Tracker;

using TrackResult = std::tuple<unsigned int, bool, bool>;

// Grain ids per rounded x coordinate of the largest segment
std::map<int, unsigned int>
ids_by_position(const Tracker<dim, Number> &tracker)
{
  std::map<int, unsigned int> result;
  for (const auto *grain : sorted_grains(tracker))
    {
      const auto &segments = grain->get_segments();
      const auto  largest =
        std::max_element(segments.begin(),
                         segments.end(),
                         [](const auto &a, const auto &b) {
                           return a.get_measure() < b.get_measure();
                         });

      result.emplace(static_cast<int>(std::round(largest->get_center()[0])),
                     grain->get_grain_id());
    }

  return result;
}

// Order parameters per rounded x coordinate of the largest segment
std::map<int, unsigned int>
order_parameters_by_position(const Tracker<dim, Number> &tracker)
{
  std::map<int, unsigned int> result;
  for (const auto *grain : sorted_grains(tracker))
    {
      const auto &segments = grain->get_segments();
      const auto  largest =
        std::max_element(segments.begin(),
                         segments.end(),
                         [](const auto &a, const auto &b) {
                           return a.get_measure() < b.get_measure();
                         });

      result.emplace(static_cast<int>(std::round(largest->get_center()[0])),
                     grain->get_order_parameter_id());
    }

  return result;
}

void
report(ConditionalOStream         &pcout,
       const std::string          &label,
       const Tracker<dim, Number> &tracker,
       const bool                  ids_preserved,
       const TrackResult          &result)
{
  const auto [n_collisions, grains_reassigned, op_number_changed] = result;

  std::set<unsigned int> order_parameters;
  for (const auto &[gid, grain] : tracker.get_grains())
    {
      (void)gid;
      order_parameters.insert(grain.get_order_parameter_id());
    }

  pcout << label << std::endl;
  pcout << "  n_grains                  = " << tracker.get_grains().size()
        << std::endl;
  pcout << "  n_collisions              = " << n_collisions << std::endl;
  pcout << "  grains_reassigned         = " << grains_reassigned << std::endl;
  pcout << "  op_number_changed         = " << op_number_changed << std::endl;
  pcout << "  n_order_parameters_in_use = " << order_parameters.size()
        << std::endl;
  pcout << "  grain ids are preserved   = " << ids_preserved << std::endl;

  for (const auto *grain : sorted_grains(tracker))
    pcout << "  grain with " << grain->n_segments() << " segment(s)"
          << std::endl;
}

void
test_prevented_collision(ConditionalOStream &pcout)
{
  pcout << "Collision across the periodic boundary is prevented" << std::endl;

  Domain domain(true);

  const unsigned int n_order_parameters = 2;

  // The grains are recognized as neighbors if the gap is smaller than 1.5
  TrackerSettings settings;
  settings.buffer_distance_fixed = 1.5;

  auto tracker = make_tracker(domain, settings);

  const std::vector<Disc> discs_before = {{Point<dim>(2., 5.), 1.5, 0},
                                          {Point<dim>(8., 5.), 1.5, 0}};

  auto solution_before = domain.make_solution(discs_before, n_order_parameters);
  solution_before.update_ghost_values();

  const auto result_before =
    tracker.initial_setup(solution_before, n_order_parameters);

  const auto reference_ids = ids_by_position(tracker);
  const auto ops           = order_parameters_by_position(tracker);

  report(pcout, "Before the growth:", tracker, true, result_before);

  /* The grains have grown, but sit in different order parameters now. The
   * solution is built accordingly, as if the grains had been remapped. */
  const std::vector<Disc> discs_after = {{Point<dim>(2., 5.), 2.1, ops.at(2)},
                                         {Point<dim>(8., 5.), 2.1, ops.at(8)}};

  auto solution_after = domain.make_solution(discs_after, n_order_parameters);
  solution_after.update_ghost_values();

  const auto result_after = tracker.track(solution_after, n_order_parameters);

  report(pcout,
         "After the growth:",
         tracker,
         ids_by_position(tracker) == reference_ids,
         result_after);

  solution_before.zero_out_ghost_values();
  solution_after.zero_out_ghost_values();
}

void
test_detected_merge(ConditionalOStream &pcout)
{
  pcout << "Merge of two grains across the periodic boundary is detected"
        << std::endl;

  Domain domain(true);

  const unsigned int n_order_parameters = 2;

  // The gap between the grains is larger than the buffer
  TrackerSettings settings;

  auto tracker = make_tracker(domain, settings);

  const std::vector<Disc> discs_before = {{Point<dim>(2., 5.), 1.5, 0},
                                          {Point<dim>(8., 5.), 1.5, 0}};

  auto solution_before = domain.make_solution(discs_before, n_order_parameters);
  solution_before.update_ghost_values();

  const auto result_before =
    tracker.initial_setup(solution_before, n_order_parameters);

  report(pcout, "Before the merge:", tracker, true, result_before);

  /* The grains have grown so much within a single step that their fields
   * are now connected across the periodic boundary. */
  const std::vector<Disc> discs_after = {{Point<dim>(2., 5.), 2.1, 0},
                                         {Point<dim>(8., 5.), 2.1, 0}};

  auto solution_after = domain.make_solution(discs_after, n_order_parameters);
  solution_after.update_ghost_values();

  bool        exception_thrown = false;
  bool        message_is_clear = false;
  std::string what;
  try
    {
      tracker.track(solution_after, n_order_parameters);
    }
  catch (const ExceptionBase &exception)
    {
      exception_thrown = true;

      // The exception wraps long lines, so the white spaces are normalized
      std::istringstream words(exception.what());
      for (std::string word; words >> word;)
        what += word + " ";

      message_is_clear =
        what.find("collided and merged") != std::string::npos &&
        what.find("no successor") != std::string::npos &&
        what.find("overlaps 2 old grains") != std::string::npos;
    }

  pcout << "After the merge:" << std::endl;
  pcout << "  exception is thrown = " << exception_thrown << std::endl;
  pcout << "  message is clear    = " << message_is_clear << std::endl;

  solution_before.zero_out_ghost_values();
  solution_after.zero_out_ghost_values();
}

void
test_boundary_crossing(ConditionalOStream &pcout)
{
  pcout << "A grain crosses the periodic boundary" << std::endl;

  Domain domain(true);

  const unsigned int n_order_parameters = 2;

  TrackerSettings settings;

  auto tracker = make_tracker(domain, settings);

  const std::vector<Disc> discs_before = {{Point<dim>(2., 5.), 1.5, 0}};

  auto solution_before = domain.make_solution(discs_before, n_order_parameters);
  solution_before.update_ghost_values();

  const auto result_before =
    tracker.initial_setup(solution_before, n_order_parameters);

  const unsigned int reference_id = tracker.get_grains().begin()->first;

  report(pcout, "Before crossing:", tracker, true, result_before);

  const std::vector<Disc> discs_after = {{Point<dim>(1.2, 5.), 1.5, 0}};

  auto solution_after = domain.make_solution(discs_after, n_order_parameters);
  solution_after.update_ghost_values();

  const auto result_after = tracker.track(solution_after, n_order_parameters);

  report(pcout,
         "After crossing:",
         tracker,
         tracker.get_grains().size() == 1 &&
           tracker.get_grains().begin()->first == reference_id,
         result_after);

  solution_before.zero_out_ghost_values();
  solution_after.zero_out_ghost_values();
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const bool is_zero_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
  ConditionalOStream pcout(std::cout, is_zero_rank);

  pcout << std::boolalpha;

  test_prevented_collision(pcout);
  pcout << std::endl;
  test_detected_merge(pcout);
  pcout << std::endl;
  test_boundary_crossing(pcout);
}
