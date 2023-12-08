//
// Created by Joshua Ji on 2023-12-07.
//

#include "CallErrorWalker.h"

#include <utility>

namespace gazprea {
    CallErrorWalker::CallErrorWalker(std::shared_ptr<SymbolTable> symTab) {
        this->symTab = std::move(symTab);
        contexts = {WALKER_CONTEXT::NONE};
    }

    std::string CallErrorWalker::debugContext() {
        std::string result = "\n\tCtxs: ";
        for (const auto& context : contexts) {
            result += contextToString(context) + " ";
        }
        return result;
    }

    bool CallErrorWalker::inContext(WALKER_CONTEXT context) {
        return std::find(contexts.begin(), contexts.end(), context) != contexts.end();
    }

    std::string CallErrorWalker::contextToString(WALKER_CONTEXT context)  {
        switch (context) {
            case WALKER_CONTEXT::FUNCTION:
                return "FUNCTION";
            case WALKER_CONTEXT::DECL_BODY:
                return "DECL_BODY";
            case WALKER_CONTEXT::VECTOR_LITERAL:
                return "VECTOR_LITERAL";
            case WALKER_CONTEXT::NONE:
                return "NONE";
        }
    }

    std::any CallErrorWalker::visitCall(std::shared_ptr<CallNode> tree) {

    }

    // === STREAM ===

    std::any CallErrorWalker::visitStreamIn(std::shared_ptr<StreamIn> tree) {
        walkChildren(tree);
    }
}
