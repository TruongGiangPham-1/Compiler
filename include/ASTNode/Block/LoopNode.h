#include "BlockNode.h"

class LoopNode : public BlockNode {
public:
    std::shared_ptr<ASTNode> condition; // could be null later on

    LoopNode(size_t tokenType, int line) : BlockNode(tokenType, line) {}
    std::string toString() override;
};
