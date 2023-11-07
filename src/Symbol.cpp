#include "Symbol.h"
#include "ASTNode/ASTNode.h"

Symbol::Symbol(std::string name) : name(name), typeSym(nullptr) {}
Symbol::Symbol(std::string name, std::shared_ptr<Type> type) : name(name), typeSym(type) {}
Symbol::Symbol(std::string name, std::shared_ptr<Type> type  , std::shared_ptr<Scope> scope) : name(name), typeSym(type), scope(scope) {}


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
