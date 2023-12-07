//
// Created by Joshua Ji on 2023-11-16.
//

#include "SyntaxWalker.h"

//#define DEBUG

namespace gazprea {
    SyntaxWalker::SyntaxWalker() {
        scopeDepth = 0;
        contexts = {CONTEXT::NONE};
        hasMain = false;
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

    std::string SyntaxWalker::debugContext() {
        std::string result = "\n\tCtxs: ";
        for (const auto& context : contexts) {
            result += contextToString(context) + " ";
        }
        return result;
    }

    bool SyntaxWalker::inContext(CONTEXT context) {
        return std::find(contexts.begin(), contexts.end(), context) != contexts.end();
    }

    std::string SyntaxWalker::contextToString(gazprea::CONTEXT context)  {
        switch (context) {
            case CONTEXT::FUNCTION:
                return "FUNCTION";
            case CONTEXT::DECL_BODY:
                return "DECL_BODY";
            case CONTEXT::VECTOR_LITERAL:
                return "VECTOR_LITERAL";
            case CONTEXT::NONE:
                return "NONE";
        }
    }

    // Declaration
    std::any SyntaxWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toStringTree()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
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
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        scopeDepth++;
        walkChildren(tree);
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitConditional(std::shared_ptr<ConditionalNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
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
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        scopeDepth++;
        walk(tree->getBody());
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        scopeDepth++;
        walk(tree->getBody());
        scopeDepth--;
        return 0;
    }

    std::any SyntaxWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
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
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
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
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
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
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif

        // a function or procedure call
        // if we are in a global declaration initialization, this is an error
        if (inContext(CONTEXT::DECL_BODY) && inGlobalScope()) {
            throw SyntaxError(tree->loc(), "Global initialization cannot contain function/procedure calls");
        }

        // if this is a procedure call and we are in a function body, this is an error
        if (inContext(CONTEXT::FUNCTION) && tree->procCall) {
            throw SyntaxError(tree->loc(), "Function body cannot contain (impure) procedure calls");
        }
        return 0;
    }

    std::any SyntaxWalker::visitVector(std::shared_ptr<VectorNode> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        // if we are already in a vector, this is an error
        //if (inContext(CONTEXT::VECTOR_LITERAL)) {
        //    throw SyntaxError(tree->loc(), "Bad vector literal (too many nested vectors)");
        //}

        //// else, we are in a vector. Go through children
        //contexts.push_back(CONTEXT::VECTOR_LITERAL);
        //walkChildren(tree);
        //contexts.pop_back();

        return 0;
    }

    // === IMPURE STATEMENTS ===

    std::any SyntaxWalker::visitStreamIn(std::shared_ptr<StreamIn> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        // if we are in a function context, this is an error
        if (inContext(CONTEXT::FUNCTION)) {
            throw SyntaxError(tree->loc(), "Function body cannot contain streamin (impure I/O)");
        }
        return 0;
    }

    std::any SyntaxWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
#ifdef DEBUG
        std::cout << "Visiting " << tree->toString()
                  << " inside global scope: " << debugGlobalScope() << debugContext() << std::endl;
#endif
        // if we are in a function context, this is an error
        if (inContext(CONTEXT::FUNCTION)) {
            throw SyntaxError(tree->loc(), "Function body cannot contain streamout (impure I/O)");
        }
        return 0;
    }
}
