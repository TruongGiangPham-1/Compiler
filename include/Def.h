//
// Created by truong on 28/10/23.
//

#ifndef GAZPREABASE_DEF_H
#define GAZPREABASE_DEF_H

#include "ASTWalker.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"
#include "CompileTimeExceptions.h"
#include "ScopedSymbol.h"
#include "AdvanceType.h"

namespace gazprea {
    class Def : public ASTWalker {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;

        std::unordered_map<std::string, std::shared_ptr<ASTNode>> forwardDeclFuncBody;
        std::unordered_map<std::string, std::shared_ptr<ASTNode>> forwardDeclProcBody;
        int getNextId();

        Def(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirID);
        void defineFunctionAndProcedureArgs(int loc, std::shared_ptr<Symbol> methodSym, std::vector<std::shared_ptr<ASTNode>>orderedArgs,
                                            std::shared_ptr<Type> retType ,int isFunc); //

        std::shared_ptr<int> varID;

        // === TOP LEVEL AST NODES ===
        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        // === EXPRESSION AST NODES ===
        std::any visitID(std::shared_ptr<IDNode> tree) override;
        // === TYPE
        std::any visitTypedef(std::shared_ptr<TypeDefNode> tree) override;

        // Expr/Vector
        //std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
        //std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;

        // === BLOCK AST NODES ===
        std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;

        std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;
        std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        //std::any visitBlock(std::shared_ptr<BlockNode>tree) override;
    };
}
#endif //GAZPREABASE_DEF_H
