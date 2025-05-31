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

#include <deal.II/base/timer.h>

void
monitor(const std::string label)
{
  // return;

  dealii::Utilities::System::MemoryStats stats;
  dealii::Utilities::System::get_memory_stats(stats);

  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "MONITOR " << label << ": ";

  if (label != "break")
    {
      const auto print = [&pcout](const double value) {
        const auto min_max_avg =
          dealii::Utilities::MPI::min_max_avg(value / 1e6, MPI_COMM_WORLD);

        pcout << min_max_avg.min << " " << min_max_avg.max << " "
              << min_max_avg.avg << " " << min_max_avg.sum << " ";
      };

      print(stats.VmPeak);
      print(stats.VmSize);
      print(stats.VmHWM);
      print(stats.VmRSS);
    }

  pcout << std::endl;
}

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
  string_to_enum(const std::string label)
  {
    if (label == "never")
      return never;
    if (label == "n_calls")
      return n_calls;
    if (label == "real_time")
      return real_time;
    if (label == "simulation_time")
      return simulation_time;

    AssertThrow(false, dealii::ExcNotImplemented());

    return never;
  }

public:
  TimerPredicate(const std::string label,
                 const double      start    = 0.0,
                 const double      interval = 0.0)
  {
    reinit(string_to_enum(label), start, interval);
  }

  TimerPredicate(const Type   type     = Type::never,
                 const double start    = 0.0,
                 const double interval = 0.0)
  {
    reinit(type, start, interval);
  }

  void
  reinit(const Type type, const double start, const double interval = 0.0)
  {
    this->counter              = 0;
    this->last_simulation_time = 0;

    this->type = type;

    if (type == Type::never)
      {
        // nothing to do
      }
    else if (type == Type::n_calls)
      {
        this->counter  = static_cast<unsigned int>(start);
        this->interval = interval;
      }
    else if (type == Type::simulation_time)
      {
        this->last_simulation_time = start;
        this->interval             = interval;
      }
    else if (type == Type::real_time)
      {
        this->last_real_time = std::chrono::system_clock::now();
        this->interval       = interval;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }
  }

  bool
  now(const double time = 0.0)
  {
    ++counter;

    if (type == Type::never)
      {
        return false;
      }
    else if (type == Type::n_calls)
      {
        if (interval <= 0.0)
          return false;

        return ((counter - 1) % static_cast<unsigned int>(interval)) == 0;
      }
    else if (type == Type::simulation_time)
      {
        if (interval <= 0.0)
          return false;

        if ((time - last_simulation_time - interval) > -1e-10)
          last_simulation_time = time;
        return last_simulation_time == time;
      }
    else if (type == Type::real_time)
      {
        if (interval <= 0.0)
          return false;

        const bool do_output =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now() - last_real_time)
              .count() /
            1e9 >
          interval;

        if (dealii::Utilities::MPI::sum<unsigned int>(do_output,
                                                      MPI_COMM_WORLD) == 0)
          return false;

        last_real_time = std::chrono::system_clock::now();
        return true;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
        return false;
      }
  }

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
  attach(const MyTimerOutput &timer)
  {
    static int counter = 0;

    get_instance().timers.emplace(counter++, &timer);
  }

  static void
  detach(const MyTimerOutput &timer)
  {
    auto &set = get_instance().timers;

    const auto ptr =
      std::find_if(set.begin(), set.end(), [&timer](const auto &p) {
        return p.second == &timer;
      });

    Assert(ptr != set.end(),
           dealii::ExcMessage(
             "Timer could not be found! Have you attached it?"));

    set.erase(ptr);
  }

  static void
  print_all_wall_time_statistics(const bool force_output = false);

  static TimerCollection &
  get_instance()
  {
    static TimerCollection instance;

    return instance;
  }

  static void
  configure(const double interval)
  {
    get_instance().predicate.reinit(TimerPredicate::real_time,
                                    0.0 /*dummy value*/,
                                    interval);
  }

private:
  TimerCollection()
  {}


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
  MyTimerOutput(const bool enabled = true)
    : pcout(std::cout,
            dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , timer(pcout, dealii::TimerOutput::never, dealii::TimerOutput::wall_times)
    , enabled(enabled)
  {
    if (enabled)
      TimerCollection::attach(*this);
  }

  ~MyTimerOutput()
  {
    if (enabled)
      TimerCollection::detach(*this);
  }


  dealii::TimerOutput &
  operator()()
  {
    return timer;
  }

  const dealii::TimerOutput &
  operator()() const
  {
    return timer;
  }

  void
  enter_subsection(const std::string label)
  {
    (void)label;
#ifdef WITH_TIMING
    timer.enter_subsection(label);
#endif
  }

  void
  leave_subsection(const std::string label)
  {
    (void)label;
#ifdef WITH_TIMING
    timer.leave_subsection(label);
#endif
  }

  void
  print_wall_time_statistics() const
  {
    Assert(enabled, dealii::ExcInternalError());

    if (timer.get_summary_data(dealii::TimerOutput::OutputData::total_wall_time)
          .size() > 0)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);
  }

  bool
  is_enabled() const
  {
    return enabled;
  }

private:
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput        timer;

  const bool enabled;
};



void
TimerCollection::print_all_wall_time_statistics(const bool force_output)
{
  if (force_output || get_instance().predicate.now())
    for (const auto &timer : get_instance().timers)
      timer.second->print_wall_time_statistics();
}



class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_,
          const std::string   &section_name,
          const bool           do_timing = true)
    : section_name(section_name)
  {
#ifdef WITH_TIMING
    if (do_timing)
      scope =
        std::make_unique<dealii::TimerOutput::Scope>(timer_, section_name);
    enter();
#else
    (void)timer_;
    (void)do_timing;
#endif
  }

  MyScope(MyTimerOutput     &timer_,
          const std::string &section_name,
          const bool         do_timing = true)
    : MyScope(timer_(), section_name, do_timing)
  {}

  MyScope(const std::string &section_name, MyTimerOutput *ptr_timer = nullptr)
    : section_name(section_name)
  {
#ifdef WITH_TIMING
    if (ptr_timer)
      scope = std::make_unique<dealii::TimerOutput::Scope>((*ptr_timer)(),
                                                           section_name);
    enter();
#else
    (void)ptr_timer;
#endif
  }

  ~MyScope()
  {
#ifdef WITH_TIMING
    leave();
#endif
  }

private:
  void
  enter()
  {
#ifdef WITH_TIMING_OUTPUT
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << ">>> MyScope: entering <" << section_name << ">"
                << std::endl;
    monitor(section_name + "_in_");
#endif
  }

  void
  leave()
  {
#ifdef WITH_TIMING_OUTPUT
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << ">>> MyScope: leaving <" << section_name << ">" << std::endl;
    monitor(section_name + "_out");
#endif
  }

  const std::string section_name;
#ifdef WITH_TIMING
  std::unique_ptr<dealii::TimerOutput::Scope> scope;
#endif
};
