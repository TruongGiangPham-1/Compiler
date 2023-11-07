#include "ASTNode/Loop/PredicatedLoopNode.h"

PredicatedLoopNode::PredicatedLoopNode(int line) : LoopNode(line) {};

std::string PredicatedLoopNode::toString() {
    return "PredicatedLoopNode";
}

std::shared_ptr<ExprNode> PredicatedLoopNode::getCondition() {
return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::shared_ptr<BlockNode> PredicatedLoopNode::getBody() {
    return std::dynamic_pointer_cast<BlockNode>(children[1]);
}
