//
// Created by truong on 01/11/23.
//
#include "../include/ScopedSymbol.h"
#include "../include/Symbol.h"



std::shared_ptr<Symbol> ScopedSymbol::resolve(const std::string &name) {
    for ( auto sym : orderedArgs ) {
        if ( sym->getName() == name ) {
            return sym;
        }
    }
    // if not here, check any enclosing scope
    if ( getEnclosingScope() != nullptr ) {
        return getEnclosingScope()->resolve(name);
    }
    return nullptr; // not found
}

std::shared_ptr<Type> ScopedSymbol::resolveType(const std::string &name) {
    // Really, there is nothing in function/procedure block to resolve type, so just ask the enclosing scope
    if ( enclosingScope != nullptr ) return enclosingScope->resolveType(name);
    return nullptr; // not found
}

void ScopedSymbol::define(std::shared_ptr<Symbol> sym) {
    orderedArgs.push_back(sym);
}


std::shared_ptr<Scope> ScopedSymbol::getEnclosingScope() { return enclosingScope; }

std::string ScopedSymbol::getName() {
    return name;
}

void ScopedSymbol::setEnclosingScope(std::shared_ptr<Scope> scope) {
    enclosingScope = scope;
}
std::string ScopedSymbol::toString() {
    std::stringstream str;
    str << "method" << Symbol::toString() << ":{";
    for (auto iter = orderedArgs.begin(); iter != orderedArgs.end(); iter++) {
        std::shared_ptr<Symbol> sym = *iter;
        if ( iter != orderedArgs.begin() ) str << ", ";
        str << sym->toString();
    }
    str << "}";
    return str.str();
}

bool ScopedSymbol::isBuiltIn() const {
    return isBuiltin;
}