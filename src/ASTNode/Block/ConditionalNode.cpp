#include "ASTNode/Block/ConditionalNode.h"

ConditionalNode::ConditionalNode(int line) : BlockNode(line) {}

std::string ConditionalNode::toString() {
    return "Conditional";
}
