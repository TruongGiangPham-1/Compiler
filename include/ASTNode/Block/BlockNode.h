#pragma once

#include "ASTNode/ASTNode.h"

// right now, this is mainly used as a parent class for loops and conditionals (a node that contains a nested scopee)
// but in the future, it can also stand by itself for Block Statements (https://cmput415.github.io/415-docs/gazprea/spec/statements.html#block-statements)
// Do we need this? Is it find to just use an ASTNode?
// not sure, but I think it's better to be more explicit
class BlockNode : public ASTNode {
public:
    BlockNode(size_t tokenType, int line);

    std::string toString() override;

    std::vector<std::shared_ptr<ASTNode>> getStatements();
};
