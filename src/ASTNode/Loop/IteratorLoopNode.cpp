//
// Created by truong on 27/11/23.
//
#include "ASTNode/Loop/IteratorLoopNode.h"

IteratorLoopNode::IteratorLoopNode(int line) : LoopNode(line){

}

std::shared_ptr<BlockNode> IteratorLoopNode::getBody() {
    auto blockNode = std::dynamic_pointer_cast<BlockNode>(this->children[children.size() - 1]);
    return blockNode;
}


std::vector<std::shared_ptr<ExprNode>> IteratorLoopNode::getConditions() {
    std::vector<std::shared_ptr<ExprNode>> ret;  //  from josh's code in vectorNode.h
    ret.reserve(children.size() - 1);
    for (int i = 0; i < children.size() - 1; i++) {  // ignore the last child cuz thats the body
        ret.push_back(std::static_pointer_cast<ExprNode>(children[i]));
    }
    return ret;
}

std::string IteratorLoopNode::toString() {
    return "iteratorNode";
}
