#pragma once
#include "../ExprNode.h"

// `val` is calculated in the first pass (Builder)
class TupleNode: public ExprNode {
public:
    std::vector<std::shared_ptr<ExprNode>> val;

    TupleNode(int line);

    std::string toString() override;
    std::vector<ExprNode> getVal();
};
