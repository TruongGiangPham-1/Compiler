# Gather our source files in this directory.
set(
  gazprea_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

# Build our executable from the source files.
add_executable(gazc ${gazprea_src_files})
target_include_directories(gazc PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(gazc antlr)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs core)

# Add the LLVM, antlr runtime and parser as libraries to link.
target_link_libraries(gazc parser antlr4-runtime ${llvm_libs})

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("gazc")
