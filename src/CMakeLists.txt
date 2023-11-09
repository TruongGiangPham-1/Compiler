# Gather our source files in this directory.
set(
  gazprea_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/BackEnd.cpp"
        ../include/Symbol.h
        ../include/Type.h
        ../include/Scope.h
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
        ASTNode/Type/TypeNode.cpp
        ../include/ASTNode/Type/TypeNode.h
        ../include/BuiltinTypes/
        ../include/ASTNode
        ASTNode/AssignNode.cpp
        ../include/ASTNode/AssignNode.h
        ASTNode/DeclNode.cpp
        ../include/ASTNode/DeclNode.h
        ../include/ASTNode/Expr/ExprNode.h
        ASTNode/Expr/Literal/IDNode.cpp
        ../include/ASTNode/Expr/Literal/IDNode.h
        ASTNode/Expr/Literal/IntNode.cpp
        ../include/ASTNode/Expr/Literal/IntNode.h
        ASTNode/Expr/Literal/RealNode.cpp
        ../include/ASTNode/Expr/Literal/RealNode.h
        ASTNode/Expr/Literal/CharNode.cpp
        ../include/ASTNode/Expr/Literal/CharNode.h
        ASTNode/Expr/Literal/TupleNode.cpp
        ../include/ASTNode/Expr/Literal/TupleNode.h
        ASTNode/Expr/Literal/BoolNode.cpp
        ../include/ASTNode/Expr/Literal/BoolNode.h
        ASTNode/Expr/Binary/BinaryExpr.cpp
        ../include/ASTNode/Expr/Binary/BinaryExpr.h
        ASTNode/Expr/Vector/RangeVecNode.cpp
        ../include/ASTNode/Expr/Vector/RangeVecNode.h
        ASTNode/Expr/Unary/UnaryExpr.cpp
        ../include/ASTNode/Expr/Unary/UnaryExpr.h
        ../include/ASTNode/Expr/Vector/BaseVectorExpr.h
        ASTNode/Block/BlockNode.cpp
        ../include/ASTNode/Block/BlockNode.h
        ASTNode/Loop/LoopNode.cpp
        ../include/ASTNode/Loop/LoopNode.h
        ASTNode/Block/ConditionalNode.cpp
        ../include/ASTNode/Block/ConditionalNode.h
        ASTNode/Expr/Vector/GeneratorNode.cpp
        ../include/ASTNode/Expr/Vector/GeneratorNode.h
        ASTNode/Expr/Vector/FilterNode.cpp
        ../include/ASTNode/Expr/Vector/FilterNode.h
        ASTNode/Stream/StreamOut.cpp
        ../include/ASTNode/Stream/StreamOut.h
        ASTWalker.cpp
        ../include/TypeWalker.h
        ../include/BackendWalker.h
        TypeWalker.cpp
        Def.cpp
        BackendWalker.cpp
        ../include/customError/ErrorListener.h
        ScopedSymbol.cpp
        ../include/ScopedSymbol.h
        ../include/ASTNode/Method/FunctionNode.h
        ASTNode/Method/FunctionNode.cpp
        ../include/Ref.h
        Ref.cpp
        ../include/ASTNode/CallNode.h
        ../include/FunctionCallTypes/FuncCallType.h
        FunctionCallTypes/FuncCallType.cpp
        ASTNode/Method/ProcedureNode.cpp
        ../include/ASTNode/Method/ProcedureNode.h
        ../include/ASTNode/Expr/NullNode.h
        ../include/ASTNode/Expr/IdentityNode.h
        ASTNode/Expr/IdentityNode.cpp
        ASTNode/Expr/NullNode.cpp
        ASTNode/Type/VectorTypeNode.cpp
        ../include/ASTNode/Type/VectorTypeNode.h
        ../include/Types/TYPES.h
        ASTNode/Type/StringTypeNode.cpp
        ../include/ASTNode/Type/StringTypeNode.h
        ASTNode/Type/MatrixTypeNode.cpp
        ../include/ASTNode/Type/MatrixTypeNode.h
        ASTNode/Type/TupleTypeNode.cpp
        ../include/ASTNode/Type/TupleTypeNode.h
        ASTNode/TypeDefNode.cpp
        ../include/ASTNode/TypeDefNode.h
        ASTNode/ArgNode.cpp
        ../include/ASTNode/ArgNode.h
        ../include/AdvanceType.h
        ASTNode/Loop/InfiniteLoopNode.cpp
        ASTNode/Loop/PredicatedLoopNode.cpp
        ASTNode/Loop/PostPredicatedLoopNode.cpp
        ASTNode/BreakNode.cpp
        ../include/ASTNode/BreakNode.h
        ASTNode/ContinueNode.cpp
        ../include/ASTNode/ContinueNode.h
        ../include/ASTNode/Expr/ExprListNode.h
        ASTNode/Expr/ExprListNode.cpp
        ASTNode/Stream/StreamIn.cpp
        ../include/ASTNode/Stream/StreamIn.h
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
