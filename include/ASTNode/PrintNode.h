#include "ASTNode.h"

class PrintNode : public ASTNode{
public:
    PrintNode(size_t type, int line);

    std::shared_ptr<ASTNode> getExpr();
};

