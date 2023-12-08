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

    bool CallErrorWalker::inContext(WALKER_CONTEXT context) {

    }
}
