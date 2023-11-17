//
// Created by Joshua Ji on 2023-11-16.
//

#include "SyntaxWalker.h"

//#define DEBUG

namespace gazprea {
    SyntaxWalker::SyntaxWalker() {
        scopeDepth = 0;
        contexts = {CONTEXT::NONE};
    }

    bool SyntaxWalker::inGlobalScope() {
        return scopeDepth == 1;
    }

    std::string SyntaxWalker::debugGlobalScope() {
        if (inGlobalScope()) {
            return "true";
        } else {
            return "false " + std::to_string(scopeDepth);
        }
    }

    CONTEXT SyntaxWalker::getCurrentContext() {
        return contexts.back();
    }

    // Declaration
    std::any SyntaxWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toStringTree()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        if (inGlobalScope()) {
            if (tree->getQualifier() == QUALIFIER::VAR) {
                throw SyntaxError(tree->loc(), "Global Variables must be declared constant");
            }

            if (!tree->getExprNode()) {
                throw SyntaxError(tree->loc(), "Global Variables must be initialized");
            }
        }

        // visit DECL body
        if (tree->getExprNode()) {
            contexts.push_back(CONTEXT::DECL_BODY);
            walk(tree->getExprNode());
            contexts.pop_back();
        }

        return 0;
    }

    // === SCOPES ===
    std::any SyntaxWalker::visitBlock(std::shared_ptr<BlockNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        scopeDepth++;
        walkChildren(tree);
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitConditional(std::shared_ptr<ConditionalNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        scopeDepth++;
        for (const auto& body : tree->bodies) {
            walk(body);
        }
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        scopeDepth++;
        walk(tree->getBody());
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        scopeDepth++;
        walk(tree->getBody());
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        scopeDepth++;
        walk(tree->getBody());
        scopeDepth--;
        return 0;
    }

    // === FUNCTION/PROCEDURE ===

    std::any SyntaxWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        if (!inGlobalScope()) {
            throw SyntaxError(tree->loc(), "(forward) function declarations found in non-global scope");
        }

        contexts.push_back(CONTEXT::FUNCTION);
        if (tree->body) {
            scopeDepth++;
            walk(tree->body);
            scopeDepth--;
        }

        // visit arguments
        for (const auto& arg : tree->orderedArgs) {
            walk(arg);
        }
        contexts.pop_back();
        return 0;
    }

    std::any SyntaxWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        if (!inGlobalScope()) {
            throw SyntaxError(tree->loc(), "(forward) procedure declaration found in non-global scope");
        }

        // if there isn't a body, this is a forward declaration
        if (tree->body) {
            scopeDepth++;
            walk(tree->body);
            scopeDepth--;
        }
        return 0;
    }

    // === EXPR ===

    std::any SyntaxWalker::visitCall(std::shared_ptr<CallNode> tree) {
        // a function or procedure call
        // if we are in a global declaration initialization, this is an error
        if (getCurrentContext() == CONTEXT::DECL_BODY && inGlobalScope()) {
            throw SyntaxError(tree->loc(), "Global initialization cannot contain function/procedure calls");
        }
        return 0;
    }

    std::any SyntaxWalker::visitVector(std::shared_ptr<VectorNode> tree) {
        // if we are already in a vector, this is an error
        if (getCurrentContext() == CONTEXT::VECTOR_LITERAL) {
            throw SyntaxError(tree->loc(), "Bad vector literal (too many nested vectors)");
        }

        // else, we are in a vector. Go through children
        contexts.push_back(CONTEXT::VECTOR_LITERAL);
        walkChildren(tree);
        contexts.pop_back();
    }
}
