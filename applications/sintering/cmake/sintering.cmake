# Customizable settings
# Define max sintring grains
set(MAX_SINTERING_GRAINS 10 CACHE STRING "Maximum number of grains")

# Possible options variants
set(VARIANTS_DIM "2D;3D" CACHE STRING "Problem dimensions")
set(VARIANTS_MOBILITY "scalar;tensorial" CACHE STRING "Mobility types")

# Mapping
set(OPTION_DIM_2D "SINTERING_DIM=2")
set(OPTION_DIM_3D "SINTERING_DIM=3")
set(OPTION_MOB_SCALAR "")
set(OPTION_MOB_TENSORIAL "WITH_TENSORIAL_MOBILITY")

message(STATUS "Maximum sintering grains: ${MAX_SINTERING_GRAINS}")

message(STATUS "Configured executables:")

macro(define_sintering_apps source_file_name executable_prefix generated_targets)

  foreach(VAR_DIM ${VARIANTS_DIM})
    foreach(VAR_MOB ${VARIANTS_MOBILITY})

      string(TOUPPER ${VAR_DIM} VAR_DIM_UP)
      string(TOUPPER ${VAR_MOB} VAR_MOB_UP)

      set(executable_name "${executable_prefix}-${VAR_DIM}-${VAR_MOB}")

      ADD_EXECUTABLE(${executable_name} "${source_file_name}")
      DEAL_II_SETUP_TARGET(${executable_name})

      TARGET_COMPILE_DEFINITIONS(${executable_name} PUBLIC
        -D${OPTION_DIM_${VAR_DIM_UP}}
        -DMAX_SINTERING_GRAINS=${MAX_SINTERING_GRAINS}
        -D${OPTION_MOB_${VAR_MOB_UP}}
      )

      TARGET_LINK_LIBRARIES(${executable_name} "pf-applications" "sintering")
      target_include_directories(${executable_name} PUBLIC "include/")

      list(APPEND ${generated_targets} ${executable_name})

      message(STATUS "  ${executable_name}")
    endforeach()
  endforeach()
endmacro()
