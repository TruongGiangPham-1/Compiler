//
// Created by Joshua Ji on 2023-11-13.
//

#include "ASTNode/Expr/Literal/VectorNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
VectorNode::VectorNode(int line) : ExprNode(line) {}

std::vector<std::shared_ptr<ExprNode>> VectorNode::getElements() {
    std::vector<std::shared_ptr<ExprNode>> ret;
    ret.reserve(children.size());
    for (auto &child : children) {
        ret.push_back(std::static_pointer_cast<ExprNode>(child));
    }

    return ret;
}

std::shared_ptr<ExprNode> VectorNode::getElement(int i) {
    return std::static_pointer_cast<ExprNode>(children[i]);
}

int VectorNode::getSize() {
    return children.size();
}

std::string VectorNode::toString() {
    return "VectorNode";
}

std::string VectorNode::getVal() {
    std::string result = "";
    if (this->isString) {
        for (auto &child : children) {
            result += (std::static_pointer_cast<CharNode>(child)->getVal());
        }
    }
    return result;
}