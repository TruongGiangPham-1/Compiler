#pragma once

#include "Symbol.h"
#include "ASTNode.h"

// a type when declaring a variable
class TypeNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type

    TypeNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ASTNode(token), sym(sym) {}
    TypeNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}

    std::string getTypeName();
    std::string toString() override;
};
