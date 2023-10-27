#include "ASTNode/Block/LoopNode.h"

LoopNode::LoopNode(int line) : BlockNode(line) {};

std::string LoopNode::toString() {
    return "Loop";
}
