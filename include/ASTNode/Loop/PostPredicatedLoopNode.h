#pragma once
#include "LoopNode.h"

class PostPredicatedLoopNode : public LoopNode {
public:
    PostPredicatedLoopNode(int line);

    std::shared_ptr<ExprNode> getCondition();
    std::shared_ptr<BlockNode> getBody();
    std::string toString() override;
};
