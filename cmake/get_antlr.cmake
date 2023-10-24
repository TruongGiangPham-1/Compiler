# A module to obtain the ANTLR C++ runtime. It attempts to find a locally installed copy of the
# runtime pointed to by an environment variable called ANTLR_INS. Can swap in get_antlr_manual
# if you'd rather have it auto-install a copy just for this project (more space).

# CMake module that finds a previously installed copy of the ANTLR C++ runtime required when linking
# a generated lexer/parser. Creates the variable ANTLR_INCLUDE_DIRS to add to your target's
# include directories, adds the antlr library path to the project, allows your target to link
# against antlr4-runtime, creates ANTLR_JAR for generating grammars, and creates an antlr target
# that you can add via add_dependencies to make sure everything has happened already.

# Get the environment variable that tells us where the manual install was.
if (NOT DEFINED ENV{ANTLR_INS})
  message(FATAL_ERROR "Did you forget to install ANTLR? The ANTLR_INS environment variable was "
                      "not set.")
endif()
file(TO_CMAKE_PATH "$ENV{ANTLR_INS}" _ANTLR_DIR)

# Set the directory for binaries.
file(TO_CMAKE_PATH "${_ANTLR_DIR}/bin" BIN_DIR) # Join dir.
set(BIN_DIR ${BIN_DIR} CACHE PATH "ANTLR jar directory.") # Set for internal use.

# Download ANTLR executable, saves us from ensuring people have java build tools (e.g. Maven)...
file(TO_CMAKE_PATH "${BIN_DIR}/antlr-4.13.0-complete.jar" ANTLR_JAR)
if (NOT EXISTS "${ANTLR_JAR}")
  message(STATUS "Downloading ANTLR generator...")
  file(
    DOWNLOAD
    http://www.antlr.org/download/antlr-4.13.0-complete.jar
    "${ANTLR_JAR}"
    SHOW_PROGRESS
  )
  file(TO_NATIVE_PATH "${BIN_DIR}" BIN_DIR_NATIVE) # Transform for display.
  message(STATUS "Downloaded ANTLR jar destination: ${BIN_DIR_NATIVE}")
endif()

# Check that the base include path exists.
if (NOT EXISTS "${_ANTLR_DIR}/include/antlr4-runtime/")
  message(FATAL_ERROR "Missing ANTLR include dir: ${_ANTLR_DIR}/include/antlr4-runtime/")
endif()

# Create a list of include dirs with the base directory.
list(APPEND ANTLR_INCLUDE_DIRS "${_ANTLR_DIR}/include/antlr4-runtime")

# Try to add the subdirectories to the include dirs.
foreach(src_path misc atn dfa tree support)
  # Check that each of the directories exists, too many files to check individually.
  if (NOT EXISTS "${_ANTLR_DIR}/include/antlr4-runtime/${src_path}")
    message(FATAL_ERROR "Missing ANTLR include dir: "
                        "${_ANTLR_DIR}/include/antlr4-runtime/${src_path}")
  endif()

  # Add the directory to the includes list.
  list(APPEND ANTLR_INCLUDE_DIRS "${_ANTLR_DIR}/include/antlr4-runtime/${src_path}")
endforeach(src_path)

# Check that the ANTLR libs exist.
set(
  _antlr_static
    "${_ANTLR_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}antlr4-runtime${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
if (NOT EXISTS "${_antlr_static}")
  message(FATAL_ERROR "Missing static ANTLR lib: ${_antlr_static}")
endif()

# Make antlr4-runtime a static library.
add_library(antlr4-runtime STATIC IMPORTED)
set_property(
  TARGET antlr4-runtime
  PROPERTY
  IMPORTED_LOCATION ${_antlr_static}
)

# Add dummy target. There's nothing to be built because we're just importing antlr and exporting
# values.
add_custom_target(antlr)
