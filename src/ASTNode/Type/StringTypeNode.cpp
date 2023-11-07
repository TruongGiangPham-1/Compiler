#include "ASTNode/Type/StringTypeNode.h"

#include <utility>

StringTypeNode::StringTypeNode(int line, std::shared_ptr<Symbol> sym) : TypeNode(line, sym) {
    this->line = line;
    this->size = nullptr;
}

StringTypeNode::StringTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> size) : TypeNode(line, sym) {
    this->line = line;
    this->size = std::move(size);
}

std::shared_ptr<ExprNode> StringTypeNode::getSize() const {
    return std::dynamic_pointer_cast<ExprNode>(size);
}

bool StringTypeNode::isInferred() const {
    return size == nullptr;
}

std::string StringTypeNode::toString() {
    return "StringTypeNode [" + size->toStringTree() + "]";
}