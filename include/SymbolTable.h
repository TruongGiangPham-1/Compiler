#pragma once

#include <memory>
#include <vector>

#include "BaseScope.h"

class SymbolTable {
private:
    std::vector<Scope*> scopes;
public:
    SymbolTable() {}
    Scope* globalScope;
    Scope* enterScope(std::string& name, Scope* currentScope);
    Scope* enterScope(Scope* newScope);

    Scope* exitScope(Scope* currentScope) {
        return currentScope->getEnclosingScope();
    }

    std::string toString();
};
