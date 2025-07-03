CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(TEST_LIBRARIES pf-applications sintering)
INCLUDE_DIRECTORIES("include" ${CMAKE_SOURCE_DIR}/applications/sintering/include ${CMAKE_SOURCE_DIR}/applications/structural/include)

FILE(GLOB_RECURSE RUNNER_TARGET_SRC "src/*.cc")
ADD_LIBRARY(runner-models ${RUNNER_TARGET_SRC})
TARGET_LINK_LIBRARIES(runner-models "pf-applications" "sintering")

DEAL_II_SETUP_TARGET(runner-models)

SET(TEST_LIBRARIES pf-applications sintering runner-models)

add_compile_definitions("SOURCE_CODE_ROOT=${CMAKE_SOURCE_DIR}")

DEAL_II_PICKUP_TESTS()

ADD_SUBDIRECTORY(include)
