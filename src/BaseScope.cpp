#include <iostream>
#include <sstream>
#include <string>

#include "ScopedSymbol.h"
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

void BaseScope::defineType(std::shared_ptr<Symbol> sym) {
    auto cast = std::dynamic_pointer_cast<AdvanceType>(sym);
    this->userTypes.emplace(cast->getTypDefname(), cast);
}

std::shared_ptr<Type> BaseScope::resolveType(const std::string &typeName) {
    auto find_s = userTypes.find(typeName);
    if ( find_s != userTypes.end() ) {
        std::string foundTypeName = find_s->second->getName();
        // keep looking up the map until we find the topmost typedef
        while (userTypes.find(foundTypeName) != userTypes.end()) {
            // we keep finding typedef mapping up the chain
            std::string temp = foundTypeName;
            find_s = userTypes.find(foundTypeName);
            foundTypeName = userTypes[foundTypeName]->getName();
            if (foundTypeName == temp) {   // eg we found <"integer", "integer"> mapping. return to avoid infinite loop
                return std::dynamic_pointer_cast<Type>(find_s->second);
            }
        }
        return std::dynamic_pointer_cast<Type>(find_s->second);
    }
    // if not here, check any enclosing scope
    if ( enclosingScope != nullptr ) return enclosingScope->resolveType(typeName);
    return nullptr; // not found
}

