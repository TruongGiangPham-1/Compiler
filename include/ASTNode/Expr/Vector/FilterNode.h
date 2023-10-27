#include "BaseVectorNode.h"

// [domainVar in vector & expr]
// children: [vector, expr]
class FilterNode : public BaseVectorNode {
public:
    std::string domainVar;
    FilterNode(size_t type, std::string domainVar, int line);

    std::shared_ptr<ASTNode> getVecNode();
    std::shared_ptr<ASTNode> getExpr();
};
