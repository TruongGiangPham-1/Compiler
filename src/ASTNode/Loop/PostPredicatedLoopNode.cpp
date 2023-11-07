#include "ASTNode/Loop/PostPredicatedLoopNode.h"

PostPredicatedLoopNode::PostPredicatedLoopNode(int line) : LoopNode(line) {};

std::string PostPredicatedLoopNode::toString() {
    return "PostPredicatedLoopNode";
}

std::shared_ptr<ExprNode> PostPredicatedLoopNode::getCondition() {
return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::shared_ptr<BlockNode> PostPredicatedLoopNode::getBody() {
    return std::dynamic_pointer_cast<BlockNode>(children[1]);
}
