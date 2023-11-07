#include "ASTNode/Loop/LoopNode.h"

LoopNode::LoopNode(int line) : ASTNode(line) {};

std::string LoopNode::toString() {
    return "Loop";
}

std::shared_ptr<ExprNode> LoopNode::getCondition() {
    return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::shared_ptr<BlockNode> LoopNode::getBody() {
    return std::dynamic_pointer_cast<BlockNode>(children[1]);
}
