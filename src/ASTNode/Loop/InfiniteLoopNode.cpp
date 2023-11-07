#include "ASTNode/Loop/InfiniteLoopNode.h"

InfiniteLoopNode::InfiniteLoopNode(int line) : LoopNode(line) {};

std::string InfiniteLoopNode::toString() {
    return "InfiniteLoopNode";
}

std::shared_ptr<BlockNode> InfiniteLoopNode::getBody() {
    return std::dynamic_pointer_cast<BlockNode>(children[0]);
}