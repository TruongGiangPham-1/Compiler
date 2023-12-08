//
// Created by Joshua Ji on 2023-12-07.
//

#include "CallErrorWalker.h"

#include <utility>

namespace gazprea {
    CallErrorWalker::CallErrorWalker(std::shared_ptr<SymbolTable> symTab) : ContextedWalker() {
        this->symTab = std::move(symTab);
    }
}
