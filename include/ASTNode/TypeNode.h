#pragma once

#include "Symbol.h"
#include "ASTNode.h"

// a type when declaring a variable
class TypeNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type

    TypeNode(int line, std::shared_ptr<Symbol> sym);

    std::string getTypeName();
    std::string toString() override;
};
