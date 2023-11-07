#pragma once

#include "ASTNode.h"

// instead of holding an ID node, we directly hold the symbol
// Children: [ ExprNode ]
class AssignNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;

    AssignNode(int line, std::shared_ptr<Symbol> sym);

    std::shared_ptr<Symbol> getID();

    std::shared_ptr<ASTNode> getLvalue();
    std::shared_ptr<ASTNode> getRvalue();


    std::string toString() override;
    std::string getIDName();
};
