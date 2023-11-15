//
// Created by Joshua Ji on 2023-11-15.
//

#include "ASTNode/Expr/Literal/MatrixNode.h"

MatrixNode::MatrixNode(int line) : ExprNode(line) {}

std::vector<std::shared_ptr<VectorNode>> MatrixNode::getElements() {
    std::vector<std::shared_ptr<VectorNode>> ret;
    ret.reserve(children.size());
    for (auto &child : children) {
        ret.push_back(std::static_pointer_cast<VectorNode>(child));
    }

    return ret;
}

std::shared_ptr<VectorNode> MatrixNode::getElement(int i) {
    return std::static_pointer_cast<VectorNode>(children[i]);
}

std::shared_ptr<ExprNode> MatrixNode::getElement(int i, int j) {
    return std::static_pointer_cast<ExprNode>(getElement(i)->getElement(j));
}

int MatrixNode::getRowSize() {
    return children.size();
}

int MatrixNode::getColSize() {
    return getElement(0)->getSize();
}
