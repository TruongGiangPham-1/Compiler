#pragma once
#include "ASTNode/ASTNode.h"

// Base class for all expressions
// Theoretically, this should not be instantiated by itself. Should I make it virtual?
class ExprNode : public ASTNode {
public:
    std::shared_ptr<Type> type;  // For type checking

    ExprNode(size_t tokenType) : ASTNode(tokenType), type(nullptr) {}
    ExprNode(antlr4::Token* token) : ASTNode(token), type(nullptr) {}
};
