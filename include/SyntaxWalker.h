//
// Created by Joshua Ji on 2023-11-16.
//

#ifndef GAZPREABASE_SYNTAXWALKER_H
#define GAZPREABASE_SYNTAXWALKER_H

#include "ContextedWalker.h"
#include "WalkerContext.h"

namespace gazprea {
class SyntaxWalker : public ContextedWalker {
private:
    // the entire file is wrapped inside a `block` scope
    // thus, it's easier to just count the scope depth
    // global scope is depth == 1
    int scopeDepth;

    bool inGlobalScope();
    std::string debugGlobalScope();

public:
    SyntaxWalker();

    // declarations
    std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

    // === SCOPES ===
    std::any visitBlock(std::shared_ptr<BlockNode> tree) override;
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

#endif // GAZPREABASE_SYNTAXWALKER_H
