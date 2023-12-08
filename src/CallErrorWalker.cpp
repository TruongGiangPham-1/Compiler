//
// Created by Joshua Ji on 2023-12-07.
//

#include "CallErrorWalker.h"

#include <utility>

namespace gazprea {
    CallErrorWalker::CallErrorWalker(std::shared_ptr<SymbolTable> symTab) : ContextedWalker() {
        this->symTab = std::move(symTab);
    }

    bool CallErrorWalker::checkCallNodeType(const std::shared_ptr<CallNode>& tree) {
        if (std::dynamic_pointer_cast<FunctionSymbol>(tree->MethodRef)) {
            return false;
        } else if (std::dynamic_pointer_cast<FunctionSymbol>(tree->MethodRef)) {
            return true;
        } else {
            throw std::runtime_error("CallErrorWalker encountered something weird. See: CallErrorWalker::isProcedure method");
        }
    }

    std::any CallErrorWalker::visitCall(std::shared_ptr<CallNode> tree) {
        // is this a function or a procedure?
        bool isProcedure = checkCallNodeType(tree);
        if (!isProcedure && tree->procCall) {
            // a call keyword is used on a function!
            throw CallError(tree->loc(), "Call statement used to call a function");
        }
    }

    // === STREAM ===
    std::any CallErrorWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
        contexts.push_back(WALKER_CONTEXT::STREAM_OUT);
        walk(tree->getExpr());
        contexts.pop_back();
    }
}
