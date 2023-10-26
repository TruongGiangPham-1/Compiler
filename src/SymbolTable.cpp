//
// Created by Joshua Ji on 2023-10-24.
//

#include "SymbolTable.h"
#include <sstream>

#include "SymbolTable.h"
#include "BaseScope.h"

Scope* SymbolTable::enterScope(std::string& name, Scope* enclosingScope) {
    Scope *newScope = new LocalScope(name, enclosingScope);
    scopes.push_back(newScope);
    return newScope;
}

Scope* SymbolTable::enterScope(Scope* newScope) {
    scopes.push_back(newScope);
    return newScope;
}

std::string SymbolTable::toString() {
    std::stringstream str;
    str << "SymbolTable {" << std::endl;
    for (auto s : scopes) {
        str << s->getScopeName() << ": " << s->toString() << std::endl;
    }
    str << "}" << std::endl;
    return str.str();
}
