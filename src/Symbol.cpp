#include "Symbol.h"
#include "AST.h"

Symbol::Symbol(std::string name) : name(name), type(nullptr) {}
Symbol::Symbol(std::string name, Type* type) : name(name), type(type) {}
Symbol::Symbol(std::string name, Type* type, Scope* scope) : name(name), type(type), scope(scope) {}


std::string Symbol::getName() { return name; }

std::string Symbol::toString() {
    if (type != nullptr) return '<' + getName() + ":" + type->getName() + '>';
    return getName();
}

std::string Symbol::source() {
    std::string info = name + " defined ";
    if (scope) {
        info += "in scope " + scope->getScopeName() + " ";
    }
    return info + "\n";
}

Symbol::~Symbol() {}
