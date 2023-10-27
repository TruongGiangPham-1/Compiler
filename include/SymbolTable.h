#pragma once

#include <memory>
#include <vector>

#include "BaseScope.h"

class SymbolTable {
private:
    std::vector<std::shared_ptr<Scope>> scopes;
public:
    SymbolTable() {}
    std::shared_ptr<Scope> globalScope;
    std::shared_ptr<Scope> enterScope(std::string& name, const std::shared_ptr<Scope>& currentScope);
    std::shared_ptr<Scope> enterScope(std::shared_ptr<Scope> newScope);

    std::shared_ptr<Scope> exitScope(std::shared_ptr<Scope> currentScope) {
        return currentScope->getEnclosingScope();
    }

    std::string toString();
};
