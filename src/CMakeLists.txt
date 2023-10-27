# Gather our source files in this directory.
set(
  gazprea_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/BackEnd.cpp"
        ../include/Symbol.h
        ../include/Type.h
        ../include/Scope.h
        ../include/BuiltInTypeSymbol.h
        BaseScope.cpp
        ../include/BaseScope.h
        ASTNode/ASTNode.cpp
        ../include/ASTNode/ASTNode.h
        ASTBuilder.cpp
        ../include/ASTBuilder.h
        Symbol.cpp
        ../include/ASTWalker.h
        SymbolTable.cpp
        ../include/SymbolTable.h
        ../include/Operands/UNARYOP.h
        ASTNode/TypeNode.cpp
        ../include/ASTNode/TypeNode.h
        ASTNode/AssignNode.cpp
        ../include/ASTNode/AssignNode.h
        ASTNode/DeclNode.cpp
        ../include/ASTNode/DeclNode.h
        ../include/ASTNode/Expr/ExprNode.h
        ASTNode/Expr/IDNode.cpp
        ../include/ASTNode/Expr/IDNode.h
        ASTNode/Expr/IntNode.cpp
        ../include/ASTNode/Expr/IntNode.h
        ASTNode/Expr/Binary/BinaryExpr.cpp
        ../include/ASTNode/Expr/Binary/BinaryExpr.h
        ASTNode/Expr/Binary/RangeVecNode.cpp
        ../include/ASTNode/Expr/Binary/RangeVecNode.h
        ASTNode/Expr/Unary/UnaryExpr.cpp
        ../include/ASTNode/Expr/Unary/UnaryExpr.h
)

# Build our executable from the source files.
add_executable(gazc ${gazprea_src_files})
target_include_directories(gazc PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(gazc antlr)

# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS Core Support)
llvm_map_components_to_libnames(llvm_libs core)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Add the MLIR, LLVM, antlr runtime and parser as libraries to link.
target_link_libraries(gazc PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
    )

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("gazc")
