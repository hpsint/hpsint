CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(TEST_LIBRARIES pf-applications)
INCLUDE_DIRECTORIES("include" ${CMAKE_SOURCE_DIR}/applications/sintering/include ${CMAKE_SOURCE_DIR}/applications/structural/include)

DEAL_II_PICKUP_TESTS()

ADD_SUBDIRECTORY(include)
