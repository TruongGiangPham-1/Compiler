#pragma once

#include "BaseVectorNode.h"

class RangeVecNode : public BaseVectorNode {
public:
    RangeVecNode(int line) : BaseVectorNode(line) {}

    std::string toString() override;

    std::shared_ptr<ASTNode> getStart();
    std::shared_ptr<ASTNode> getEnd();
};
