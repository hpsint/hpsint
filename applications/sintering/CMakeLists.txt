# Sintering 2D, max 10 particles, tensorial mobility
ADD_EXECUTABLE(sintering-2D-tensor "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-2D-tensor)
TARGET_COMPILE_DEFINITIONS(sintering-2D-tensor PUBLIC -DSINTERING_DIM=2 -DMAX_SINTERING_GRAINS=10 -DWITH_TENSORIAL_MOBILITY)
TARGET_LINK_LIBRARIES(sintering-2D-tensor "pf-applications")
target_include_directories(sintering-2D-tensor PUBLIC "include/" "../structural/include/")

# Sintering 2D, max 10 particles, scalar mobility
ADD_EXECUTABLE(sintering-2D-scalar "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-2D-scalar)
TARGET_COMPILE_DEFINITIONS(sintering-2D-scalar PUBLIC -DSINTERING_DIM=2 -DMAX_SINTERING_GRAINS=10)
TARGET_LINK_LIBRARIES(sintering-2D-scalar "pf-applications")
target_include_directories(sintering-2D-scalar PUBLIC "include/" "../structural/include/")

# Sintering 3D, max 10 particles, tensorial mobility
ADD_EXECUTABLE(sintering-3D-tensor "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-3D-tensor)
TARGET_COMPILE_DEFINITIONS(sintering-3D-tensor PUBLIC -DSINTERING_DIM=3 -DMAX_SINTERING_GRAINS=10 -DWITH_TENSORIAL_MOBILITY)
TARGET_LINK_LIBRARIES(sintering-3D-tensor "pf-applications")
target_include_directories(sintering-3D-tensor PUBLIC "include/" "../structural/include/")

# Sintering 3D, max 10 particles, scalar mobility
ADD_EXECUTABLE(sintering-3D-scalar "sintering.cc")
DEAL_II_SETUP_TARGET(sintering-3D-scalar)
TARGET_COMPILE_DEFINITIONS(sintering-3D-scalar PUBLIC -DSINTERING_DIM=3 -DMAX_SINTERING_GRAINS=10)
TARGET_LINK_LIBRARIES(sintering-3D-scalar "pf-applications")
target_include_directories(sintering-3D-scalar PUBLIC "include/" "../structural/include/")