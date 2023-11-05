#pragma once
#include "../ExprNode.h"

// `val` is calculated in the first pass (Builder)
class BoolNode : public ExprNode {
public:
    bool val;

    BoolNode(int line, bool val);

    std::string toString() override;
    bool getVal();
};
