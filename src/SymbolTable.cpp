//
// Created by Joshua Ji on 2023-10-24.
//

#include "SymbolTable.h"
#include <sstream>

#include "SymbolTable.h"
#include "BaseScope.h"
#include "ASTNode/Type/TypeNode.h"

std::shared_ptr<Scope> SymbolTable::enterScope(std::string& name, const std::shared_ptr<Scope>& enclosingScope) {
    std::shared_ptr<Scope> newScope = std::make_shared<LocalScope>(name, enclosingScope);
    scopes.push_back(newScope);
    return newScope;
}

std::shared_ptr<Scope> SymbolTable::enterScope(std::shared_ptr<Scope> newScope) {
    scopes.push_back(newScope);
    return newScope;
}

std::string SymbolTable::toString() {
    std::stringstream str;
    str << "SymbolTable {" << std::endl;
    for (const auto& s : scopes) {
        str << s->getScopeName() << ": " << s->toString() << std::endl;
    }
    str << "}" << std::endl;
    return str.str();
}

TYPE SymbolTable::resolveType(std::shared_ptr<ASTNode> typeNode) {
    std::shared_ptr<TypeNode> typeN = std::dynamic_pointer_cast<TypeNode>(typeNode);
    std::cout << "Resolve Type: " << typeN->getTypeName() << std::endl;
    if (typeN == nullptr) {
        throw TypeError(typeNode->loc(), "cannot cast to TypeNode");
    }
    return typeN->typeEnum;
}