#pragma once

class MyScope
{
public:
  MyScope(dealii::TimerOutput &timer_, const std::string &section_name)
#ifdef WITH_TIMING
    : scope(timer_, section_name)
#endif
  {
    (void)timer_;
    (void)section_name;
  }

  ~MyScope() = default;

private:
#ifdef WITH_TIMING
  dealii::TimerOutput::Scope scope;
#endif
};
