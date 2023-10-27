#include "ExprNode.h"

// `val` is calculated in the first pass (Builder)
class IntNode : public ExprNode {
public:
    int val;

    IntNode(size_t tokenType, int val) : ExprNode(tokenType), val(val) {}

    std::string toString() override;
    int getVal();
};
