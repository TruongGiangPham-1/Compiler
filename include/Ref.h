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
#include "FunctionCallTypes/FuncCallType.h"


namespace gazprea {
    class Ref: public ASTWalker {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;

        std::shared_ptr<Type> resolveType(std::shared_ptr<ASTNode> t);

        int getNextId();
        void defineFunctionAndProcedure(int loc, std::shared_ptr<Symbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                          int isFunc); //
        Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr);

        std::shared_ptr<int> varID;

        // === EXPRESSION AST NODES ===
        std::any visitID(std::shared_ptr<IDNode> tree) override;

        // === BlOCK FUNCTION AST NODES ===
        std::any visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree) override;
        std::any visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) override;


        // === Function Call ===
        std::any visitFunction_call(std::shared_ptr<FunctionCallNode> tree) override;
    };
}
#endif //GAZPREABASE_REF_H
