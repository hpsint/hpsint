CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_P4EST) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_P4EST = ON
It was set to OFF in your installation but is required for this program."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_QUERY_GIT_INFORMATION(PF_APPLICATIONS)

SET(TARGET_SRC inst.cc)
FILE(GLOB_RECURSE ADDITIONAL_TARGET_SRC "src/*.cc")

PROJECT(pf-applications)

ADD_LIBRARY(pf-applications ${TARGET_SRC} ${ADDITIONAL_TARGET_SRC})

install(TARGETS pf-applications LIBRARY DESTINATION lib)

ADD_SUBDIRECTORY(include)

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(pf-applications)

target_include_directories(pf-applications PUBLIC "include/" ${CMAKE_CURRENT_BINARY_DIR}/include)

OPTION(USE_SNES "Use SNES (experimantal)." OFF)
OPTION(DEAL_II_WITH_USE_SNES ${USE_SNES})

IF(${USE_SNES})
  target_compile_definitions(pf-applications PUBLIC USE_SNES)
ENDIF()

FILE(GLOB SOURCE_FILES "*.cc")

# Customizable settings
# Hints for likwid
set(LIKWID_LIB CACHE STRING "Path to likwid library")
set(LIKWID_INC CACHE STRING "Path to likwid includes")

option(TARGET_AC "Build Allen-Cahn executables" ON)
option(TARGET_CH "Build Cahn-Hilliard executables" ON)
option(TARGET_GP "Build grand potential executables" ON)
option(TARGET_SINTERING "Build sintering executables" ON)
option(TARGET_GRAIN_GROWTH "Build grain growth executables" ON)
option(TARGET_GP_GREENQUIST "Build Greenquist grand potential model executables" ON)
option(TARGET_STRUCTURAL "Build structural executables" ON)
option(TARGET_BENCHMARKS "Build benchmarks" ON)
option(TARGET_TESTS "Build tests" ON)

FIND_LIBRARY(LIKWID likwid HINTS $ENV{LIKWID_LIB} ${LIKWID_LIB})

IF(LIKWID)
  message(STATUS "Likwid library detected in path: " ${LIKWID})
ENDIF()

if(TARGET_AC)
  message(STATUS "Building allen_cahn targets")
  ADD_SUBDIRECTORY(applications/allen_cahn)
ENDIF()

if(TARGET_CH)
  message(STATUS "Building cahn_hilliard targets")
  ADD_SUBDIRECTORY(applications/cahn_hilliard)
ENDIF()

if(TARGET_GP)
  message(STATUS "Building grand_potential targets")
  ADD_SUBDIRECTORY(applications/grand_potential)
ENDIF()

# This is a complex submodule, it checks the targets itself
message(STATUS "Building sintering targets")
ADD_SUBDIRECTORY(applications/sintering)
ADD_SUBDIRECTORY(applications/sintering/include)
ADD_SUBDIRECTORY(applications/structural/include)

if(TARGET_STRUCTURAL)
  message(STATUS "Building structural targets")
  ADD_SUBDIRECTORY(applications/structural)
ENDIF()

if(TARGET_BENCHMARKS)
  message(STATUS "Building benchmarks targets")
  ADD_SUBDIRECTORY(benchmarks)
ENDIF()

if(TARGET_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  message(STATUS "Building tests targets")
  set(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)

  enable_testing()
  ADD_SUBDIRECTORY(tests)
endif()
