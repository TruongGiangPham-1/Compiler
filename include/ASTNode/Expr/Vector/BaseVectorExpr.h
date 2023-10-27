#pragma once
#include "ASTNode/ASTNode.h"

// should never be instantiated by itself
class BaseVectorExpr : public ASTNode {
public:
    BaseVectorExpr(int line) : ASTNode(line) {}
};

