#pragma once
#include "ASTNode.h"
#include "ASTNode/Type/TypeNode.h"

// Children : [ TypeNode }
class TypeDefNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;
    TypeDefNode(int line, std::shared_ptr<Symbol> sym);

    std::string getName();
    std::shared_ptr<TypeNode> getType();

    std::string toString() override;
};
