#pragma once
#include "BlockNode.h"

class LoopNode : public ASTNode {
public:
    std::shared_ptr<ASTNode> condition; 
    std::shared_ptr<ASTNode> body;
    LoopNode(int line);
    std::string toString() override;
};
