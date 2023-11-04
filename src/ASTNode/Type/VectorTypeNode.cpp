#include "ASTNode/Type/VectorTypeNode.h"

#include <utility>

VectorTypeNode::VectorTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> innerType) : TypeNode(line, sym) {
    this->line = line;
    this->size = nullptr;
    this->innerType = std::move(innerType);
}

VectorTypeNode::VectorTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> size, std::shared_ptr<ASTNode> innerType) : TypeNode(line, sym) {
    this->line = line;
    this->size = std::move(size);
    this->innerType = std::move(innerType);
}

std::shared_ptr<ExprNode> VectorTypeNode::getSize() const {
    return std::dynamic_pointer_cast<ExprNode>(size);
}

std::shared_ptr<TypeNode> VectorTypeNode::getInnerType() const {
    return std::dynamic_pointer_cast<TypeNode>(innerType);
}

bool VectorTypeNode::isInferred() const {
    return size == nullptr;
}

std::string VectorTypeNode::toString() {
    return "VectorTypeNode " + sym->getName() + " <" + innerType->toString() + ">";
}