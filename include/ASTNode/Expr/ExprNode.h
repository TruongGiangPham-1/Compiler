#pragma once
#include "ASTNode/ASTNode.h"
#include <string>

// Base class for all expressions
// Theoretically, this should not be instantiated by itself
class ExprNode : public ASTNode {
public:
    std::shared_ptr<Type> promoteTo;
    std::shared_ptr<Type> type;  // For type checking

    ExprNode(int line) : ASTNode(line), type(nullptr) {}
};
