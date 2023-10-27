#pragma once

#include "BaseVectorExpr.h"

class RangeVecNode : public BaseVectorExpr {
public:
    RangeVecNode(int line) : BaseVectorExpr(line) {}

    std::string toString() override;

    std::shared_ptr<ASTNode> getStart();
    std::shared_ptr<ASTNode> getEnd();
};
