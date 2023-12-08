//
// Created by Joshua Ji on 2023-11-16.
//

#ifndef GAZPREABASE_SYNTAXWALKER_H
#define GAZPREABASE_SYNTAXWALKER_H

#include "ASTWalker.h"
#include "WalkerContext.h"

namespace gazprea {
    class SyntaxWalker : public ASTWalker {
    private:
        // the entire file is wrapped inside a `block` scope
        // thus, it's easier to just count the scope depth
        // global scope is depth == 1
        int scopeDepth;

        bool inGlobalScope();
        std::string debugGlobalScope();

        // how many layers of a vector literal are we in?
        int getVectorLiteralDepth();

        // WALKER_CONTEXT gives us more info as to what we're currently visiting
        // it's a vector so it's easy to push/pop as we enter into new contexts
        std::vector<WALKER_CONTEXT> contexts;
        bool inContext(WALKER_CONTEXT context);
        std::string debugContext();

        static std::string contextToString(WALKER_CONTEXT context);
    public:
        SyntaxWalker();

        // declarations
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        // === SCOPES ===
        std::any visitBlock(std::shared_ptr<BlockNode>tree) override;
        std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
        std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) override;
        std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) override;
        std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) override;

        std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
        std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;

        // === EXPR ===
        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        std::any visitVector(std::shared_ptr<VectorNode> tree) override;

        // === IMPURE STATEMENTS ===
        std::any visitStreamIn(std::shared_ptr<StreamIn> tree) override;
        std::any visitStreamOut(std::shared_ptr<StreamOut> tree) override;
        // this also includes visitCall when it is a procedure call
    };

}


#endif //GAZPREABASE_SYNTAXWALKER_H
