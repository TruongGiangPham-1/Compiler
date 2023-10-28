#pragma once

#include "ASTNode.h"

class PrintNode : public ASTNode{
public:
    PrintNode(int line);

    std::shared_ptr<ASTNode> getExpr();

    std::string toString() override;
};

