#pragma once

#include "Symbol.h"
#include "ASTNode/ASTNode.h"
#include "Types/TYPES.h"

// a type when declaring a variable
// Unlike most "parent" nodes, this one is instantiated!
// when we have a simple type (integer, real, boolean), we instantiate TypeNode
// for more complex types (integer[*]), we use the children nodes
class TypeNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type
    // enum corresponding to the type
    // if it is a TYPE::STRING, we can dyamically cast to StringTypeNode
    TYPE typeEnum;

    TypeNode(int line, std::shared_ptr<Symbol> sym);

    std::string getTypeName();
    std::string toString() override;
};
