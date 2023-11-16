//
// Created by Joshua Ji on 2023-11-16.
//

#include "SyntaxWalker.h"

//#define DEBUG

namespace gazprea {
    SyntaxWalker::SyntaxWalker() {
        scopeDepth = 0;
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

    std::any SyntaxWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        if (tree->body) {
            scopeDepth++;
            walk(tree->body);
            scopeDepth--;
        }
        return 0;
    }

    std::any SyntaxWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << std::endl;
#endif
        // if there isn't a body, this is a forward declaration
        if (tree->body) {
            scopeDepth++;
            walk(tree->body);
            scopeDepth--;
        }
        return 0;
    }
}
