#pragma once
#include "ASTNode/Block/BlockNode.h"
#include "ASTNode/Expr/ExprNode.h"


class LoopNode : public ASTNode {
public:
    std::shared_ptr<ExprNode> condition; 
    std::shared_ptr<BlockNode> body;
    LoopNode(int line);
    std::string toString() override;
};
