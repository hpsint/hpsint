
# Sintering 2D, max 10 particles (cloud)
ADD_EXECUTABLE(sintering-2D "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-2D)
TARGET_COMPILE_DEFINITIONS(sintering-2D PUBLIC -DSINTERING_DIM=2 -DMAX_SINTERING_GRAINS=10)
TARGET_LINK_LIBRARIES(sintering-2D "pf-applications")
target_include_directories(sintering-2D PUBLIC "include/")

# Sintering 3D, max 10 particles (cloud)
ADD_EXECUTABLE(sintering-3D "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-3D)
TARGET_COMPILE_DEFINITIONS(sintering-3D PUBLIC -DSINTERING_DIM=3 -DMAX_SINTERING_GRAINS=10)
TARGET_LINK_LIBRARIES(sintering-3D "pf-applications")
target_include_directories(sintering-3D PUBLIC "include/")