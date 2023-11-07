#pragma once
#include "LoopNode.h"

class PredicatedLoopNode : public LoopNode {
public:
    PredicatedLoopNode(int line);

    std::shared_ptr<ExprNode> getCondition();
    std::shared_ptr<BlockNode> getBody();
    std::string toString() override;
};
