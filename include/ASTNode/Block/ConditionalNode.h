#include "BlockNode.h"

class ConditionalNode : public BlockNode {
public:
    std::shared_ptr<ASTNode> condition; // could be null later on

    ConditionalNode(size_t tokenType, int line) : BlockNode(tokenType, line) {}
    std::string toString() override;
};
