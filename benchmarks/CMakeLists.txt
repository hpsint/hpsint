

FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )
  
  GET_FILENAME_COMPONENT(file_name ${source_file} NAME)
  
  # determine if LIKWID is requested (if the file name contains .likwid)
  STRING( FIND ${file_name} ".likwid" LIKWID_TO_BE_USED)
  
  IF(${LIKWID_TO_BE_USED})
    STRING( REPLACE ".likwid" "" file_name ${file_name} )
  ENDIF()
  
  STRING( REPLACE ".cc" "" exec ${file_name} )
  ADD_EXECUTABLE( ${exec} ${source_file})
  DEAL_II_SETUP_TARGET(${exec})
  TARGET_LINK_LIBRARIES(${exec} "pf-applications")
  
  # add LIKWID to test if requested
  IF(${LIKWID_TO_BE_USED})
    FIND_LIBRARY(LIKWID likwid HINTS $ENV{LIKWID_LIB})
    
    IF(LIKWID)
      TARGET_COMPILE_DEFINITIONS(${exec} PUBLIC LIKWID_PERFMON)
      TARGET_INCLUDE_DIRECTORIES(${exec} PUBLIC $ENV{LIKWID_INCLUDE})
      TARGET_LINK_LIBRARIES(${exec} ${LIKWID})
    ENDIF()
  ENDIF()
  
ENDFOREACH ( source_file ${SOURCE_FILES} )
