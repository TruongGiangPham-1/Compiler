#pragma once

#include "Symbol.h"
#include "ASTNode.h"

// a type when declaring a variable
class TypeNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type

    TypeNode(size_t tokenType, int line, std::shared_ptr<Symbol> sym) : ASTNode(tokenType, line), sym(sym) {}

    std::string getTypeName();
    std::string toString() override;
};
