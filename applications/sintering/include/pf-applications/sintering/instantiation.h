#pragma once

// clang-format off
#define EXPAND_OPERATIONS_N_COMP_NT(OPERATION)                         \
  constexpr int max_components = MAX_SINTERING_GRAINS + 2;             \
  AssertThrow(n_comp_nt >= 2,                                          \
              ExcMessage("Number of components " +                     \
                         std::to_string(n_comp_nt) +                   \
                         " is not precompiled!"));                     \
  AssertThrow(n_comp_nt <= max_components +2,                          \
              ExcMessage("Number of components " +                     \
                         std::to_string(n_comp_nt) +                   \
                         " is not precompiled!"));                     \
                                                                       \
  switch (n_comp_nt)                                                   \
    {                                                                  \
      case  2: { OPERATION(std::min( 2, max_components), -1); break; } \
      case  3: { OPERATION(std::min( 3, max_components), -1); break; } \
      case  4: { OPERATION(std::min( 4, max_components), -1); break; } \
      case  5: { OPERATION(std::min( 5, max_components), -1); break; } \
      case  6: { OPERATION(std::min( 6, max_components), -1); break; } \
      case  7: { OPERATION(std::min( 7, max_components), -1); break; } \
      case  8: { OPERATION(std::min( 8, max_components), -1); break; } \
      case  9: { OPERATION(std::min( 9, max_components), -1); break; } \
      case 10: { OPERATION(std::min(10, max_components), -1); break; } \
      case 11: { OPERATION(std::min(11, max_components), -1); break; } \
      case 12: { OPERATION(std::min(12, max_components), -1); break; } \
      case 13: { OPERATION(std::min(13, max_components), -1); break; } \
      case 14: { OPERATION(std::min(14, max_components), -1); break; } \
      default:                                                         \
        AssertThrow(false,                                             \
                    ExcMessage("Number of components " +               \
                               std::to_string(n_comp_nt) +             \
                               " is not precompiled!"));               \
    }
// clang-format on