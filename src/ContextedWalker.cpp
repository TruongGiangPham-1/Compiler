//
// Created by Joshua Ji on 2023-12-07.
//

#include "ContextedWalker.h"
namespace gazprea {
    ContextedWalker::ContextedWalker() {
        contexts = {WALKER_CONTEXT::NONE};
    }

    int ContextedWalker::countContextDepth(WALKER_CONTEXT ctx) {
        int count = 0;
        for (const auto &context: contexts) {
            if (context == ctx) count++;
        }
        return count;
    }

    std::string ContextedWalker::debugContext() {
        std::string result = "\n\tCtxs: ";
        for (const auto &context: contexts) {
            result += contextToString(context) + " ";
        }
        return result;
    }

    bool ContextedWalker::inContext(WALKER_CONTEXT context) {
        return std::find(contexts.begin(), contexts.end(), context) != contexts.end();
    }

    bool ContextedWalker::directlyInContext(WALKER_CONTEXT context) {
        // check if we are *directly* inside a context
        assert(!contexts.empty()); // this HAS to be true;
        return contexts.back() == context;
    }

    std::string ContextedWalker::contextToString(WALKER_CONTEXT context) {
        switch (context) {
            case WALKER_CONTEXT::FUNCTION:
                return "FUNCTION";
            case WALKER_CONTEXT::PROCEDURE:
                return "PROCEDURE";
            case WALKER_CONTEXT::DECL_BODY:
                return "DECL_BODY";
            case WALKER_CONTEXT::ASSIGN_BODY:
                return "ASSIGN_BODY";
            case WALKER_CONTEXT::VECTOR_LITERAL:
                return "VECTOR_LITERAL";
            case WALKER_CONTEXT::BINOP:
                return "BINOP";
            case WALKER_CONTEXT::NONE:
                return "NONE";
            case WALKER_CONTEXT::STREAM_OUT:
                return "STREAM_OUT";
            case WALKER_CONTEXT::INPUT_ARGS:
                return "INPUT_ARGS";
            case WALKER_CONTEXT::RETURN_STMT:
                return "RETURN_STMT";
            case WALKER_CONTEXT::CONDITIONAL_EXPR:
                return "CONDITIONAL_EXPR";
            case WALKER_CONTEXT::ITERATOR_DOMAIN:
                return "ITERATOR_DOMAIN";
        }
    }
}