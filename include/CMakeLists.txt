## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2012 - 2022 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------

#
# Configure config.h and revision.h
#
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/pf-applications/base/revision.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/pf-applications/base/revision.h
)

# Install headers
install(DIRECTORY pf-applications DESTINATION include COMPONENT library FILES_MATCHING PATTERN "*.h")

# Install generated headers
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pf-applications  DESTINATION include COMPONENT library FILES_MATCHING PATTERN "*.h")
