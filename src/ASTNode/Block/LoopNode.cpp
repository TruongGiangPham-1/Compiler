#include "ASTNode/Block/LoopNode.h"

LoopNode::LoopNode(int line) : ASTNode(line) {};

std::string LoopNode::toString() {
    return "Loop";
}
