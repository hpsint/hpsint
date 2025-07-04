CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

INCLUDE_DIRECTORIES("include" ${CMAKE_SOURCE_DIR}/applications/sintering/include ${CMAKE_SOURCE_DIR}/applications/structural/include)

FILE(GLOB_RECURSE RUNNER_TARGET_SRC "src/*.cc")
ADD_LIBRARY(runner-models ${RUNNER_TARGET_SRC})
TARGET_LINK_LIBRARIES(runner-models "pf-applications" "sintering")

DEAL_II_SETUP_TARGET(runner-models)

SET(TEST_LIBRARIES pf-applications sintering runner-models)

add_compile_definitions("SOURCE_CODE_ROOT=${CMAKE_SOURCE_DIR}")

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DebugRelease")
  SET(OUTPUT_FOLDER_SUFFIX "debug")
ELSE()
  SET(OUTPUT_FOLDER_SUFFIX "release")
ENDIF()
add_compile_definitions("OUTPUT_FOLDER_SUFFIX=${OUTPUT_FOLDER_SUFFIX}")

# Suppress additional MPI output
IF(${DEAL_II_MPIEXEC_PREFLAGS})
  STRING(APPEND DEAL_II_MPIEXEC_PREFLAGS " --quiet")
ELSE()
  SET(DEAL_II_MPIEXEC_PREFLAGS "--quiet")
ENDIF()

DEAL_II_PICKUP_TESTS()

ADD_SUBDIRECTORY(include)
