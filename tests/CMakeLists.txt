CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(TEST_LIBRARIES pf-applications)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/applications/sintering/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/applications/structural/include)

add_compile_definitions("SOURCE_ROOT=${CMAKE_SOURCE_DIR}")

DEAL_II_PICKUP_TESTS()
