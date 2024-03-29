#pragma once
#include "ASTNode/Block/BlockNode.h"
#include "ASTNode/Expr/ExprNode.h"


// Children: [ Condition?, Body ]
// this class should not be instantiated by itself!
class LoopNode : public ASTNode {
public:
    LoopNode(int line);
    std::string toString() override;

    std::shared_ptr<ExprNode> getCondition();
    std::shared_ptr<BlockNode> getBody();
};
