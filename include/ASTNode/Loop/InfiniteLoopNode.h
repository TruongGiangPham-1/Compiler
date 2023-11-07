#pragma once
#include "LoopNode.h"

class InfiniteLoopNode : public LoopNode {
  // data class.
public:
    InfiniteLoopNode(int line);

    std::shared_ptr<BlockNode> getBody();
    std::string toString() override;
};
