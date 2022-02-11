

FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )

  GET_FILENAME_COMPONENT(file_name ${source_file} NAME)

  IF(NOT _cmp_dummy AND NOT _cmp_sintering_circle AND NOT _cmp_sintering_cloud)
    STRING( REPLACE ".cc" "" exec ${file_name} )
    ADD_EXECUTABLE( ${exec} ${source_file})
    DEAL_II_SETUP_TARGET(${exec})
    TARGET_LINK_LIBRARIES(${exec} "pf-applications")
  ENDIF()

ENDFOREACH ( source_file ${SOURCE_FILES} )