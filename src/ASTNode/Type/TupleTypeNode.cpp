#include "ASTNode/Type/TupleTypeNode.h"

TupleTypeNode::TupleTypeNode(int line, std::shared_ptr<Symbol> sym) : TypeNode(line, sym) {
    innerTypes = std::vector<std::pair<std::string, std::shared_ptr<ASTNode>>>();
}

int TupleTypeNode::numElements() {
    return innerTypes.size();
}

std::vector<std::shared_ptr<TypeNode>> TupleTypeNode::getTypes() {
    std::vector<std::shared_ptr<TypeNode>> typeNodes = std::vector<std::shared_ptr<TypeNode>>();
    for (const auto& pair : innerTypes) {
        std::shared_ptr<ASTNode> node = pair.second;
        typeNodes.push_back(std::dynamic_pointer_cast<TypeNode>(node));
    }
    return typeNodes;
}

std::shared_ptr<TypeNode> TupleTypeNode::findType(int index) {
    return std::dynamic_pointer_cast<TypeNode>(innerTypes[index].second);
}

std::shared_ptr<TypeNode> TupleTypeNode::findType(const std::string& id) {
    for (const auto& pair : innerTypes) {
        if (pair.first == id) {
            return std::dynamic_pointer_cast<TypeNode>(pair.second);
        }
    }
    return nullptr;
}

std::string TupleTypeNode::toString() {
    std::string str = "TupleTypeNode (";
    for (const auto& node : innerTypes) {
        if (!node.first.empty()) {
            str += node.first + ": ";
        }
        str += node.second->toString() + ", ";
    }
    str += ")";
    return str;
}
