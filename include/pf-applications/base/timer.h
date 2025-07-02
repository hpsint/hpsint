// ---------------------------------------------------------------------
//
// Copyright (C) 2023-2025 by the hpsint authors
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

#include <deal.II/base/timer.h>

void
monitor(const std::string label);

class MyTimerOutput;

class TimerPredicate
{
public:
  enum Type
  {
    never,
    n_calls,
    real_time,
    simulation_time
  };

private:
  Type
  string_to_enum(const std::string label);

public:
  TimerPredicate(const std::string label,
                 const double      start    = 0.0,
                 const double      interval = 0.0);

  TimerPredicate(const Type   type     = Type::never,
                 const double start    = 0.0,
                 const double interval = 0.0);

  void
  reinit(const Type type, const double start, const double interval = 0.0);

  bool
  now(const double time = 0.0);

private:
  Type                                  type;
  unsigned int                          counter              = 0;
  double                                interval             = 0;
  double                                last_simulation_time = 0;
  std::chrono::system_clock::time_point last_real_time;
};

class TimerCollection
{
public:
  static void
  attach(const MyTimerOutput &timer);

  static void
  detach(const MyTimerOutput &timer);

  static void
  print_all_wall_time_statistics(const bool force_output = false);

  static TimerCollection &
  get_instance();

  static void
  configure(const double interval);

private:
  TimerCollection();

  std::set<std::pair<unsigned int, const MyTimerOutput *>> timers;
  TimerPredicate                                           predicate;

public:
  TimerCollection(const TimerCollection &) = delete;
  void
  operator=(const TimerCollection &) = delete;
};



class MyTimerOutput
{
public:
  MyTimerOutput(const bool enabled = true);

  MyTimerOutput(std::ostream &out, const bool enabled = true);

  ~MyTimerOutput();


  dealii::TimerOutput &
  operator()();

  const dealii::TimerOutput &
  operator()() const;

  void
  enter_subsection(const std::string label);

  void
  leave_subsection(const std::string label);

  void
  print_wall_time_statistics() const;

  bool
  is_enabled() const;

private:
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput        timer;

  const bool enabled;
};

class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_,
          const std::string   &section_name,
          const bool           do_timing = true);

  MyScope(MyTimerOutput     &timer_,
          const std::string &section_name,
          const bool         do_timing = true);

  MyScope(const std::string &section_name, MyTimerOutput *ptr_timer = nullptr);

  ~MyScope();

private:
  void
  enter();

  void
  leave();

  const std::string section_name;
#ifdef WITH_TIMING
  std::unique_ptr<dealii::TimerOutput::Scope> scope;
#endif
};
