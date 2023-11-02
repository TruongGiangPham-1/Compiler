//
// Created by truong on 02/11/23.
//

#ifndef GAZPREABASE_REF_H
#define GAZPREABASE_REF_H


#include "ASTWalker.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"
#include "CompileTimeExceptions.h"
#include "ScopedSymbol.h"


namespace gazprea {
    class Ref: public ASTWalker {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;

        std::shared_ptr<Type> resolveType(std::shared_ptr<ASTNode> t);

        int getNextId();

        Ref(std::shared_ptr<SymbolTable> symTab);

        int varID = 1;

        // === BlOCK FUNCTION AST NODES ===
        std::any visitFunctionForward(std::shared_ptr<FunctionForwardNode> tree) override;
        std::any visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree) override;
        std::any visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) override;
    };
}
#endif //GAZPREABASE_REF_H
