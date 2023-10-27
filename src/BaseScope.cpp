#include <iostream>
#include <sstream>
#include <string>

#include "BaseScope.h"

std::shared_ptr<Symbol> BaseScope::resolve(const std::string &name) {
    auto find_s = symbols.find(name);
    if ( find_s != symbols.end() ) return find_s->second;
    // if not here, check any enclosing scope
    if ( enclosingScope != nullptr ) return enclosingScope->resolve(name);

    return nullptr; // not found
}

void BaseScope::define(std::shared_ptr<Symbol> sym) {
    symbols.emplace(sym->name, sym);
}

std::shared_ptr<Scope> BaseScope::getEnclosingScope() {
    return enclosingScope;
}

void BaseScope::setEnclosingScope(std::shared_ptr<Scope> scope) {
    enclosingScope = scope;
}

std::string BaseScope::toString() {
    std::stringstream str;
    str << "Scope " << getScopeName() << " { ";
    for (auto iter = symbols.begin(); iter != symbols.end(); iter++) {
        std::shared_ptr<Symbol> sym = iter->second;
        if ( iter != symbols.begin() ) str << ", ";
        str << sym->toString();
    }
    str << " }";
    return str.str();
}
