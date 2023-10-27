#pragma once

#include "BinaryExpr.h"

class RangeVecNode : public BinaryExpr {
public:
    RangeVecNode(size_t tokenType) : BinaryExpr(tokenType) {}

    // we don't want to use the default BinaryExpr toString
    std::string toString() override;
};
