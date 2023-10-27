#pragma once
#include "BlockNode.h"

class LoopNode : public BlockNode {
public:
    std::shared_ptr<ASTNode> condition; // could be null later on

    LoopNode(int line);
    std::string toString() override;
};
