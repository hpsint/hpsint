# Define max sintring grains
set(MAX_SINTERING_GRAINS 10)

# Possible options variants
set(VARIANTS_DIM      "2D;3D")
set(VARIANTS_MOBILITY "scalar;tensorial")
set(VARIANTS_OPERATOR "generic;wang;diffusion")

# Mapping
set(OPTION_DIM_2D "SINTERING_DIM=2")
set(OPTION_DIM_3D "SINTERING_DIM=3")
set(OPTION_MOB_SCALAR "")
set(OPTION_MOB_TENSORIAL "WITH_TENSORIAL_MOBILITY")
set(OPTION_OP_GENERIC "OPERATOR=1")
set(OPTION_OP_WANG "OPERATOR=2")
set(OPTION_OP_DIFFUSION "OPERATOR=3")

message(STATUS "Maximum sintering grains: ${MAX_SINTERING_GRAINS}")

message(STATUS "Configured executables:")
foreach(VAR_DIM ${VARIANTS_DIM})
  foreach(VAR_MOB ${VARIANTS_MOBILITY})
    foreach(VAR_OP ${VARIANTS_OPERATOR})

      string(TOUPPER ${VAR_DIM} VAR_DIM_UP)
      string(TOUPPER ${VAR_MOB} VAR_MOB_UP)
      string(TOUPPER ${VAR_OP} VAR_OP_UP)

      set(executable_name "sintering-${VAR_DIM}-${VAR_OP}-${VAR_MOB}")

      ADD_EXECUTABLE(${executable_name} "sintering.cc")
      DEAL_II_SETUP_TARGET(${executable_name})

      TARGET_COMPILE_DEFINITIONS(${executable_name} PUBLIC
          -D${OPTION_DIM_${VAR_DIM_UP}}
          -DMAX_SINTERING_GRAINS=${MAX_SINTERING_GRAINS}
          -D${OPTION_MOB_${VAR_MOB_UP}}
          -D${OPTION_OP_${VAR_OP_UP}}
      )

      TARGET_LINK_LIBRARIES(${executable_name} "pf-applications")
      target_include_directories(${executable_name} PUBLIC "include/" "../structural/include/")

      message(STATUS "  ${executable_name}")
    endforeach()
  endforeach()
endforeach()


### Throughput application
set(executable_name "sintering-throughput")
ADD_EXECUTABLE(${executable_name} "sintering_throughput.cc")
DEAL_II_SETUP_TARGET(${executable_name})

TARGET_COMPILE_DEFINITIONS(${executable_name} PUBLIC
    -D SINTERING_DIM=3
    -D MAX_SINTERING_GRAINS=${MAX_SINTERING_GRAINS}
    -D${OPTION_MOB_SCALAR}
    -D${OPTION_OP_GENERIC}
)

TARGET_LINK_LIBRARIES(${executable_name} "pf-applications")
target_include_directories(${executable_name} PUBLIC "include/" "../structural/include/")
