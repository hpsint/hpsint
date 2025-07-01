# Sintering library
SET(TARGET_SRC dummy.cc)
FILE(GLOB_RECURSE ADDITIONAL_TARGET_SRC "src/*.cc")
ADD_LIBRARY(sintering ${TARGET_SRC} ${ADDITIONAL_TARGET_SRC})
TARGET_LINK_LIBRARIES(sintering "pf-applications")
DEAL_II_SETUP_TARGET(sintering)
target_include_directories(sintering PUBLIC "include/" "../structural/include/")
install(TARGETS sintering LIBRARY DESTINATION lib)

include(cmake/sintering.cmake)

if(TARGET_SINTERING)
  define_sintering_apps("sintering_generic.cc" "sintering-generic" generated_executables)
ENDIF()

if(TARGET_GP_GREENQUIST)
  define_sintering_apps("grand_potential_greenquist.cc" "sintering-gp" generated_executables)
ENDIF()

if(TARGET_GRAIN_GROWTH)
  define_sintering_apps("grain_growth.cc" "grain-growth" generated_executables)
ENDIF()

option(BENCHMARK_THROUGHPUT "Build throughput benchmarks" ON)
option(BENCHMARK_GT "Build grain tracker benchmarks" ON)
option(PRINT_PARTICLES "Build print particles executable" ON)

### Throughput application
IF(BENCHMARK_THROUGHPUT)
  message(STATUS "Configured throughputs:")
  foreach(VAR_MOB ${VARIANTS_MOBILITY})

    string(TOUPPER ${VAR_MOB} VAR_MOB_UP)

    set(executable_name "sintering-throughput-${VAR_MOB}")
    ADD_EXECUTABLE(${executable_name} "sintering_throughput.cc")
    DEAL_II_SETUP_TARGET(${executable_name})

    message(STATUS "  ${executable_name}")

    TARGET_COMPILE_DEFINITIONS(${executable_name} PUBLIC
        -D SINTERING_DIM=3
        -D MAX_SINTERING_GRAINS=${MAX_SINTERING_GRAINS}
        -D${OPTION_MOB_${VAR_MOB_UP}}
    )

    TARGET_LINK_LIBRARIES(${executable_name} "pf-applications")
    target_include_directories(${executable_name} PUBLIC "include/" "../structural/include/")
      
    IF(LIKWID)
      TARGET_COMPILE_DEFINITIONS(${executable_name} PUBLIC LIKWID_PERFMON)
      TARGET_INCLUDE_DIRECTORIES(${executable_name} PUBLIC $ENV{LIKWID_INCLUDE} $ENV{LIKWID_BASE}/include ${LIKWID_INC})
      TARGET_LINK_LIBRARIES(${executable_name} ${LIKWID})
    ENDIF()
  endforeach()
ENDIF()

### Print particles application
IF(PRINT_PARTICLES)
  set(print_exe "sintering-print-particles")
  message(STATUS "Configured print particles executable:")
  message(STATUS "  ${print_exe}")
  ADD_EXECUTABLE(${print_exe} "sintering_print_particles.cc")
  DEAL_II_SETUP_TARGET(${print_exe})
  TARGET_LINK_LIBRARIES(${print_exe} "pf-applications")
  target_include_directories(${print_exe} PUBLIC "include/" "../structural/include/")
ENDIF()

### Grain tracker benchmark
IF(BENCHMARK_GT)
  message(STATUS "Configured grain tracker benchmarks:")
  foreach(VAR_DIM ${VARIANTS_DIM})
    set(gt_executable_name "grain-tracker-throughput-${VAR_DIM}")
    ADD_EXECUTABLE(${gt_executable_name} "grain_tracker_throughput.cc")
    DEAL_II_SETUP_TARGET(${gt_executable_name})
    message(STATUS "  ${gt_executable_name}")

    TARGET_COMPILE_DEFINITIONS(${gt_executable_name} PUBLIC
      -D${OPTION_DIM_${VAR_DIM}}
    )

    TARGET_LINK_LIBRARIES(${gt_executable_name} "pf-applications")
    target_include_directories(${gt_executable_name} PUBLIC "include/")
  endforeach()
ENDIF()

# Install cmake macros
install(DIRECTORY cmake DESTINATION share COMPONENT scripts FILES_MATCHING PATTERN "*.cmake")
