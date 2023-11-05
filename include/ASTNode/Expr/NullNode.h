#pragma once
#include "ExprNode.h"

// `val` is calculated in the first pass (Builder)
class NullNode : public ExprNode {
public:
    NullNode(int line);

    std::string toString() override;
    int getVal();
};
