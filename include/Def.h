//
// Created by truong on 28/10/23.
//

#ifndef GAZPREABASE_DEF_H
#define GAZPREABASE_DEF_H

#include "ASTWalker.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"

namespace gazprea {
    class Def : public ASTWalker {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;

        std::shared_ptr<Type> resolveType(std::shared_ptr<ASTNode> t);

        int getNextId();

        Def(std::shared_ptr<SymbolTable> symTab);

        int varID = 1;

        // === TOP LEVEL AST NODES ===
        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;

        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        std::any visitPrint(std::shared_ptr<PrintNode> tree) override;

        std::any visitType(std::shared_ptr<TypeNode> tree) override;

        // === EXPRESSION AST NODES ===
        std::any visitID(std::shared_ptr<IDNode> tree) override;

        std::any visitInt(std::shared_ptr<IntNode> tree) override;

        // Expr/Vector
        std::any visitFilter(std::shared_ptr<FilterNode> tree) override;

        std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;

        std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;

        // === BLOCK AST NODES ===
        std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;

        std::any visitLoop(std::shared_ptr<LoopNode> tree) override;
    };

}
#endif //GAZPREABASE_DEF_H
