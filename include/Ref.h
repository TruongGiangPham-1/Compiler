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
#include "ASTNode/Type/TupleTypeNode.h"
#include "ASTNode/Type/TypeNode.h"


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
        void defineFunctionAndProcedureArgs(int loc, std::shared_ptr<ScopedSymbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                          std::shared_ptr<Type> retType); //
        // given 2 types, throw errors if they are different
        void parametersTypeCheck(std::shared_ptr<Type> typ1, std::shared_ptr<Type> type2, int loc);

        // do neccesary parameter checks for method prototype and defintion paramters
        void methodParamErrorCheck(std::vector<std::shared_ptr<ASTNode>>prototypeArg, std::vector<std::shared_ptr<ASTNode>>methodArg, int loc);

        // swap the body of the prototype and function definition to bring definition to the highest line number
        void swapMethodBody(int prototypeLine, int methodDefinitionLine, std::shared_ptr<ASTNode>prototypeNode
        , std::shared_ptr<ASTNode> methodDefTree);

        Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr);

        std::shared_ptr<int> varID;

        std::any visitTupleIndex(std::shared_ptr<TupleIndexNode> tree);

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

        // Loops and conditionals
        std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
        std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) override;
        std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) override;
        std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) override;


        std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
        std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
        // miscaleous function
        void printTupleType(std::shared_ptr<Type> ty);

    };
}
#endif //GAZPREABASE_REF_H
