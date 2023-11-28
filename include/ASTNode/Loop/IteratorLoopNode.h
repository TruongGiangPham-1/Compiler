#pragma once
#include "LoopNode.h"
#include "Symbol.h"

class IteratorLoopNode : public LoopNode {
  // data class.
public:
    std::vector<std::shared_ptr<Symbol>> domainVars;
    IteratorLoopNode(int line);

    std::vector<std::shared_ptr<ExprNode>> getConditions();
    std::shared_ptr<BlockNode> getBody();
    std::string toString() override;
};
