# Configure revision.h
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/pf-applications/base/revision.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/pf-applications/base/revision.h
)

# Install headers
install(DIRECTORY pf-applications DESTINATION include COMPONENT library FILES_MATCHING PATTERN "*.h")

# Install generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pf-applications  DESTINATION include COMPONENT library FILES_MATCHING PATTERN "*.h")
