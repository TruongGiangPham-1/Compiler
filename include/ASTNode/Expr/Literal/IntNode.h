#pragma once
#include "../ExprNode.h"

// `val` is calculated in the first pass (Builder)
class IntNode : public ExprNode {
public:
    int val;

    IntNode(int line, int val);

    std::string toString() override;
    int getVal();
};
