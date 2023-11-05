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

        int getNextId();
        void defineFunctionAndProcedure(int loc, std::shared_ptr<Symbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                          int isFunc); //
        Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr);

        std::shared_ptr<int> varID;


        // === EXPRESSION AST NODES ===
         std::any visitID(std::shared_ptr<IDNode> tree) override;

        // === BlOCK FUNCTION AST NODES ===
        std::any visitBlock(std::shared_ptr<BlockNode> tree) override;
        std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;
        std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;

        // === Function Call ===
        // === procedure
        
        //std::any visitProcedureForward(std::shared_ptr<ProcedureForwardNode> tree) override;
        //std::any visitProcedureBlock(std::shared_ptr<ProcedureBlockNode> tree) override;

    };
}
#endif //GAZPREABASE_REF_H
