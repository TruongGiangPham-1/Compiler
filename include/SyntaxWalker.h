//
// Created by Joshua Ji on 2023-11-16.
//

#ifndef GAZPREABASE_SYNTAXWALKER_H
#define GAZPREABASE_SYNTAXWALKER_H

#include "ASTWalker.h"

namespace gazprea {

    class SyntaxWalker : public ASTWalker {
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
        virtual std::any visitDecl(std::shared_ptr<DeclNode> tree);

        // === SCOPES ===
        virtual std::any visitBlock(std::shared_ptr<BlockNode>tree);
        virtual std::any visitConditional(std::shared_ptr<ConditionalNode> tree);
        virtual std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree);
        virtual std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree);
        virtual std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree);

        virtual std::any visitFunction(std::shared_ptr<FunctionNode> tree);
        virtual std::any visitProcedure(std::shared_ptr<ProcedureNode> tree);
    };

}


#endif //GAZPREABASE_SYNTAXWALKER_H
