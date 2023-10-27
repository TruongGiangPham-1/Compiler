#include "BaseVectorNode.h"

// [domainVar in vector | expr]
// children: [vector, expr]
class GeneratorNode : public BaseVectorNode {
public:
    std::string domainVar;
    GeneratorNode(size_t type, std::string domainVar, int line);

    std::shared_ptr<ASTNode> getVecNode();
    std::shared_ptr<ASTNode> getExpr();
};
