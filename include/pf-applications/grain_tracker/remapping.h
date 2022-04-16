#pragma once

namespace GrainTracker
{
  struct Remapping
  {
    unsigned int grain_id;
    unsigned int from;
    unsigned int to;

    Remapping(const unsigned int grain_id,
              const unsigned int from,
              const unsigned int to)
      : grain_id(grain_id)
      , from(from)
      , to(to)
    {}

    bool
    operator==(const Remapping &rhs) const
    {
      return grain_id == rhs.grain_id && from == rhs.from && to == rhs.to;
    }

    bool
    operator!=(const Remapping &rhs) const
    {
      return grain_id != rhs.grain_id || from != rhs.from || to != rhs.to;
    }
  };
} // namespace GrainTracker
