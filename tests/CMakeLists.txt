CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET(TEST_LIBRARIES pf-applications)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/applications/sintering/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/applications/structural/include)
LINK_LIBRARIES(${executable_name} "boost_mpi" "boost_serialization" "boost_graph_parallel")
DEAL_II_PICKUP_TESTS()
