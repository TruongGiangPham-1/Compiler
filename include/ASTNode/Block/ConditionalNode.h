#pragma once
#include "BlockNode.h"

class ConditionalNode : public BlockNode {
public:
    std::shared_ptr<ASTNode> condition; // could be null later on

    ConditionalNode(int line);
    std::string toString() override;
};
