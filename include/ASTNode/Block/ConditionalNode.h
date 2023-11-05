#pragma once
#include "BlockNode.h"
#include "../Expr/ExprNode.h"


class ConditionalNode : public ASTNode {
public:
    std::vector<std::shared_ptr<ASTNode>> conditions; // could be null later on
    
    // possibility of multiple blocks within one node
    std::vector<std::shared_ptr<ASTNode>> bodies;

    ConditionalNode(int line);
    std::string toString() override;
};
