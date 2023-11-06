#pragma once

#include "ASTNode/ASTNode.h"

// right now, this is mainly used as a parent class for loops and conditionals (a node that contains a nested scope)
// but in the future, it can also stand by itself for Block Statements (https://cmput415.github.io/415-docs/gazprea/spec/statements.html#block-statements)
class BlockNode : public ASTNode {
public:
    BlockNode(int line);

    std::string toString() override;
};
