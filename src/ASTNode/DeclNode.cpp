#include "ASTNode/DeclNode.h"
#include "ASTNode/Expr/ExprNode.h"
#include "ASTNode/Type/TypeNode.h"

DeclNode::DeclNode(int line, std::shared_ptr<Symbol> sym) : ASTNode(line), sym(sym) {}

std::string DeclNode::toString() {
    return "DECLARE";
}

std::shared_ptr<Symbol> DeclNode::getID() {
    return sym;
}

std::string DeclNode::getIDName() {
    return sym->getName();
}

std::shared_ptr<ASTNode> DeclNode::getTypeNode() {
    if (!children.empty() and std::dynamic_pointer_cast<TypeNode>(children[0])) {
        return children[0];
    }
    return nullptr;
}

std::shared_ptr<ASTNode> DeclNode::getExprNode() {
    if (children.size() == 1) {
        // might be inferred type expr
        if (std::dynamic_pointer_cast<ExprNode>(children[0])) {
            return children[0];
        }
        return nullptr;
    } else {
        if (children.size() == 2)
            return children[1];
        else
            return nullptr;
    }
}

QUALIFIER DeclNode::getQualifier() {
    return qualifier;
}
