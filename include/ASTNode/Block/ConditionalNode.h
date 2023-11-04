#pragma once
#include "BlockNode.h"

class ConditionalNode : public ASTNode {
public:
    std::shared_ptr<ASTNode> condition; // could be null later on
    std::shared_ptr<ASTNode> body;

    ConditionalNode(int line);
    std::string toString() override;
};
