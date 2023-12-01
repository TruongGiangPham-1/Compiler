#pragma once
#include "LoopNode.h"
#include "Symbol.h"

class IteratorLoopNode : public LoopNode {
  // data class.
public:
    std::vector<std::pair<std::shared_ptr<Symbol>, std::shared_ptr<ExprNode>>> domainExprs;
    IteratorLoopNode(int line);

    std::vector<std::pair<std::shared_ptr<Symbol>, std::shared_ptr<ExprNode>>> getDomainExprs();
    std::shared_ptr<BlockNode> getBody();
    std::string toString() override;
};
