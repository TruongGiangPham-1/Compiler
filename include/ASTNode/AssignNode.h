#pragma once

#include "ASTNode.h"

// Children: [ ExprListNode, ExprNode ]
class AssignNode : public ASTNode {
public:
    AssignNode(int line);

    std::shared_ptr<ASTNode> getLvalue();
    std::shared_ptr<ASTNode> getRvalue();

    std::string toString() override;
};
