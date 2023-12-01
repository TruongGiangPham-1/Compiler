//
// Created by truong on 27/11/23.
//
#include "ASTNode/Loop/IteratorLoopNode.h"

IteratorLoopNode::IteratorLoopNode(int line) : LoopNode(line) {}

std::shared_ptr<BlockNode> IteratorLoopNode::getBody() {
    auto blockNode = std::dynamic_pointer_cast<BlockNode>(this->children[0]);
    return blockNode;
}

std::vector<std::pair<std::shared_ptr<Symbol>, std::shared_ptr<ExprNode>>> IteratorLoopNode::getDomainExprs() {
    return domainExprs;
}

std::string IteratorLoopNode::toString() {
    std::string ret = "IteratorLoopNode (";

    int i = 0;
    for (auto& domainExpr : domainExprs) {
        i++;
        ret += domainExpr.first->toString() + " in " + domainExpr.second->toString();
        if (i < domainExprs.size()) {
            ret += ", ";
        }
    }

    ret += ")";
    return ret;
}
