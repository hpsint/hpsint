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

SET( TARGET_SRC dummy.cc)

PROJECT(pf-applications)

ADD_LIBRARY(pf-applications  ${TARGET_SRC})

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

FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )

  GET_FILENAME_COMPONENT(file_name ${source_file} NAME)

  string(COMPARE EQUAL "${file_name}" "dummy.cc" _cmp_dummy)
  string(COMPARE EQUAL "${file_name}" "sintering_impl.cc" _cmp_sintering)

  IF(NOT _cmp_dummy AND NOT _cmp_sintering)
    STRING( REPLACE ".cc" "" exec ${file_name} )
    ADD_EXECUTABLE( ${exec} ${source_file})
    DEAL_II_SETUP_TARGET(${exec})
    TARGET_LINK_LIBRARIES(${exec} "pf-applications")
  ENDIF()

ENDFOREACH ( source_file ${SOURCE_FILES} )

# Sintering 2D
ADD_EXECUTABLE(sintering-2D "sintering_impl.cc")
DEAL_II_SETUP_TARGET(sintering-2D)
TARGET_COMPILE_DEFINITIONS("sintering-2D" PUBLIC -DSINTERING_DIM=2)
TARGET_LINK_LIBRARIES(sintering-2D "pf-applications")

# Sintering 3D
ADD_EXECUTABLE(sintering-3D "sintering_impl.cc")
DEAL_II_SETUP_TARGET(sintering-3D)
TARGET_COMPILE_DEFINITIONS("sintering-3D" PUBLIC -DSINTERING_DIM=3)
TARGET_LINK_LIBRARIES(sintering-3D "pf-applications")
