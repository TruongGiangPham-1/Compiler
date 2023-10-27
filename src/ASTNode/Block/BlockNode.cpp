#include "ASTNode/Block/BlockNode.h"


BlockNode::BlockNode(size_t tokenType, int line) : ASTNode(tokenType, line) {}

std::string BlockNode::toString() {
    return "Block";
}

std::vector<std::shared_ptr<ASTNode>> BlockNode::getStatements() {
    return children;
}