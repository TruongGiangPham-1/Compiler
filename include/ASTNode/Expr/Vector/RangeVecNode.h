#pragma once

#include "BaseVectorNode.h"

class RangeVecNode : public BaseVectorNode {
public:
    RangeVecNode(size_t tokenType, int line) : BaseVectorNode(tokenType, line) {}

    std::string toString() override;

    std::shared_ptr<ASTNode> getStart();
    std::shared_ptr<ASTNode> getEnd();
};
