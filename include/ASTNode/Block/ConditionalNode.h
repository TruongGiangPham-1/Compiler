#pragma once
#include "BlockNode.h"
#include "../Expr/ExprNode.h"


class ConditionalNode : public ASTNode {
public:
    std::vector<std::shared_ptr<ExprNode>> conditions; // could be null later on
    
    // possibility of multiple blocks within one node
    std::vector<std::shared_ptr<BlockNode>> bodies;

    ConditionalNode(int line);
    std::string toString() override;
};
