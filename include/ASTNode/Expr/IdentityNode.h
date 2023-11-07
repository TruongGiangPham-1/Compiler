#pragma once
#include "ExprNode.h"

// `val` is calculated in the first pass (Builder)
class IdentityNode : public ExprNode {
public:
    IdentityNode(int line);

    std::string toString() override;
    int getVal();
};
