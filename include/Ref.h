//
// Created by truong on 02/11/23.
//

#ifndef GAZPREABASE_REF_H
#define GAZPREABASE_REF_H


#include "ASTWalker.h"
#include "SymbolTable.h"
#include "CompileTimeExceptions.h"
#include "ScopedSymbol.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "AdvanceType.h"


namespace gazprea {
    class Ref: public ASTWalker {
    public:
        /*
         * as stan suggested. whenenever i see a method prototype, I will store it in thse maps
         * Whenever I see a method definition that appears after its method prototype, I will swap the bodies with the prototype to move it up higher
         * in file
         *
         */
        std::unordered_map<std::string, std::shared_ptr<FunctionNode>> funcProtypeList;  // map forwad declared function prototype  for swapping
        std::unordered_map<std::string, std::shared_ptr<ProcedureNode>> procProtypeList;  // map forwad declared function prototype for swapping


        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;

        int getNextId();
        void defineFunctionAndProcedureArgs(int loc, std::shared_ptr<Symbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                          std::shared_ptr<Type> retType ,int isFunc); //
        void defineForwardFunctionAndProcedureArgs(int loc, std::shared_ptr<ScopedSymbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                            std::shared_ptr<Type> retType ); //
        void parametersTypeCheck(std::shared_ptr<Type> typ1, std::shared_ptr<Type> type2, int loc);
        Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr);

        std::shared_ptr<int> varID;


        // === EXPRESSION AST NODES ===
        std::any visitID(std::shared_ptr<IDNode> tree) override;
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;
//        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;

        // === BlOCK FUNCTION AST NODES ===
        //std::any visitBlock(std::shared_ptr<BlockNode> tree) override;
        std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        // === procedure
        std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;

        //std::any visitProcedureForward(std::shared_ptr<ProcedureForwardNode> tree) override;
        //std::any visitProcedureBlock(std::shared_ptr<ProcedureBlockNode> tree) override;

        // Loop
        std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;


        // miscaleous function
        void printTupleType(std::shared_ptr<Type> ty);

    };
}
#endif //GAZPREABASE_REF_H
