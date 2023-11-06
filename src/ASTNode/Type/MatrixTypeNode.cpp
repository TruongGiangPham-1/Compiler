#include "ASTNode/Type/MatrixTypeNode.h"

MatrixTypeNode::MatrixTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> innerType) : TypeNode(line, sym) {
    this->line = line;
    this->sizeLeft = nullptr;
    this->sizeRight = nullptr;
    this->innerType = std::move(innerType);
}

std::shared_ptr<TypeNode> MatrixTypeNode::getInnerType() const {
    return std::dynamic_pointer_cast<TypeNode>(innerType);
}

std::shared_ptr<ExprNode> MatrixTypeNode::getLeftSize() const {
    return std::dynamic_pointer_cast<ExprNode>(sizeLeft);
}

std::shared_ptr<ExprNode> MatrixTypeNode::getRightSize() const {
    return std::dynamic_pointer_cast<ExprNode>(sizeRight);
}

std::string MatrixTypeNode::toString() {
    return "MatrixTypeNode [" + sizeLeft->toStringTree() + ", " + sizeRight->toStringTree() + "] " + innerType->toString();
}