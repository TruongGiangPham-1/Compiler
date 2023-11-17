//
// Created by Joshua Ji on 2023-11-16.
//

#ifndef GAZPREABASE_SYNTAXWALKER_H
#define GAZPREABASE_SYNTAXWALKER_H

#include "ASTWalker.h"

namespace gazprea {
    // additional context as to what we're currently visiting
    enum class CONTEXT {
        FUNCTION,
        DECL_BODY, // inside `type qualifier ID = ***`
        VECTOR_LITERAL, // inside a VectorNode
        NONE,
    };

    class SyntaxWalker : public ASTWalker {
    private:
        // the entire file is wrapped inside a `block` scope
        // thus, it's easier to just count the scope depth
        // global scope is depth == 1
        int scopeDepth;

        bool inGlobalScope();
        std::string debugGlobalScope();

        // CONTEXT gives us more info as to what we're currently visiting
        // it's a vector so it's easy to push/pop
        std::vector<CONTEXT> contexts;
        CONTEXT getCurrentContext();
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

    };

}


#endif //GAZPREABASE_SYNTAXWALKER_H
