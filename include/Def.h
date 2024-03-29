//
// Created by truong on 28/10/23.
//

#ifndef GAZPREABASE_DEF_H
#define GAZPREABASE_DEF_H

#include "ASTWalker.h"
#include "AdvanceType.h"
#include "CompileTimeExceptions.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ScopedSymbol.h"
#include "SymbolTable.h"

namespace gazprea {
class Def : public ASTWalker {
    void defineBuiltins();

public:
    std::shared_ptr<SymbolTable> symtab;
    std::shared_ptr<Scope> currentScope;

    int getNextId();

    Def(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int> mlirID);

    std::shared_ptr<int> varID;

    // === TOP LEVEL AST NODES ===
    std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
    std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

    // === EXPRESSION AST NODES ===
    std::any visitID(std::shared_ptr<IDNode> tree) override;
    // === TYPE
    std::any visitTypedef(std::shared_ptr<TypeDefNode> tree) override;

    // Expr/Vector
    // std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
    // std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;

    // === BLOCK AST NODES ===
    std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;
    std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
    std::any visitCall(std::shared_ptr<CallNode> tree) override;
    // std::any visitBlock(std::shared_ptr<BlockNode>tree) override;
};
}
#endif // GAZPREABASE_DEF_H
