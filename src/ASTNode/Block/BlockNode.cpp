#include "ASTNode/Block/BlockNode.h"


BlockNode::BlockNode(int line) : ASTNode(line) {}

std::string BlockNode::toString() {
    return "Block";
}
