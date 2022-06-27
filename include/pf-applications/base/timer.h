#pragma once

#include <deal.II/base/timer.h>

class MyTimerOutput;

class TimerCollection
{
public:
  static void
  attach(const class MyTimerOutput &timer)
  {
    get_instance().timers.insert(&timer);
  }

  static void
  detach(const class MyTimerOutput &timer)
  {
    get_instance().timers.erase(&timer);
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
    auto &instance     = get_instance();
    instance.interval  = interval;
    instance.last_time = std::chrono::system_clock::now();
  }

private:
  TimerCollection()
  {}


  std::set<const MyTimerOutput *> timers;

  double                                interval;
  std::chrono::system_clock::time_point last_time;

public:
  TimerCollection(TimerCollection const &) = delete;
  void
  operator=(TimerCollection const &) = delete;
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
  print_wall_time_statistics() const
  {
    Assert(enabled, ExcInternalError());

    if (timer.get_summary_data(dealii::TimerOutput::OutputData::total_wall_time)
          .size() > 0)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);
  }

private:
  dealii::ConditionalOStream pcout;
  dealii::TimerOutput        timer;

  const bool enabled;
};



void
TimerCollection::print_all_wall_time_statistics(const bool force_output)
{
  bool do_output = force_output;

  if (do_output == false)
    do_output = get_instance().interval == 0.0;

  if (do_output == false && (get_instance().interval != -1.0))
    do_output = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now() - get_instance().last_time)
                    .count() /
                  1e9 >
                get_instance().interval;

  if (do_output == false)
    return;

  get_instance().last_time = std::chrono::system_clock::now();

  for (const auto &timer : get_instance().timers)
    timer->print_wall_time_statistics();
}



class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_,
          const std::string &  section_name,
          const bool           do_timing = true)
  {
#ifdef WITH_TIMING
    if (do_timing)
      scope =
        std::make_unique<dealii::TimerOutput::Scope>(timer_, section_name);
#else
    (void)timer_;
    (void)section_name;
    (void)do_timing;
#endif
  }

  MyScope(MyTimerOutput &    timer_,
          const std::string &section_name,
          const bool         do_timing = true)
    : MyScope(timer_(), section_name, do_timing)
  {}

  ~MyScope() = default;

private:
#ifdef WITH_TIMING
  std::unique_ptr<dealii::TimerOutput::Scope> scope;
#endif
};
