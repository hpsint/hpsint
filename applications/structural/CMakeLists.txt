FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )

  GET_FILENAME_COMPONENT(file_name ${source_file} NAME)

  STRING( REPLACE ".cc" "" exec ${file_name} )
  ADD_EXECUTABLE( ${exec} ${source_file})
  DEAL_II_SETUP_TARGET(${exec})
  TARGET_LINK_LIBRARIES(${exec} "pf-applications")
  target_include_directories(${exec} PUBLIC "include/" "../sintering/include/")

ENDFOREACH ( source_file ${SOURCE_FILES} )