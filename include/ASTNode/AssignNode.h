#pragma once

#include "ASTNode.h"

// instead of holding an ID node, we directly hold the symbol
// Children: [ ExprNode ]
class AssignNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;

    AssignNode(size_t tokenType, int line, std::shared_ptr<Symbol> sym) : ASTNode(tokenType, line), sym(sym) {}

    std::shared_ptr<Symbol> getID();
    std::shared_ptr<ASTNode> getExprNode();

    std::string toString() override;
    std::string getIDName();
};