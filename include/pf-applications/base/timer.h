#pragma once

class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_, const std::string &section_name, const bool do_timing = true)
  {
#ifdef WITH_TIMING
    if(do_timing)
      scope = std::make_unique<dealii::TimerOutput::Scope>(timer_, section_name);
#else
    (void)timer_;
    (void)section_name;
    (void)do_timing;
#endif
  }

  ~MyScope() = default;

private:
#ifdef WITH_TIMING
  std::unique_ptr<dealii::TimerOutput::Scope> scope;
#endif
};
