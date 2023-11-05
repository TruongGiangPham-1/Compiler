#include "Symbol.h"
#include "ASTNode/ASTNode.h"

Symbol::Symbol(std::string name) : name(name), type(TYPE::INTEGER) {}
Symbol::Symbol(std::string name, TYPE type) : name(name), type(type) {}
Symbol::Symbol(std::string name, TYPE type, std::shared_ptr<Scope> scope) : name(name), type(type), scope(scope) {}


std::string Symbol::getName() { return name; }

std::string Symbol::toString() {
    //if (type != nullptr) return '<' + getName() + ":" + type->getName() + '>';
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
