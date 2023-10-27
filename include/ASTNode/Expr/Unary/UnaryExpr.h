#include "ASTNode/Expr/ExprNode.h"

// Children: [expr]
class UnaryExpr : public ExprNode
{
public:
    UNARYOP op;

    UnaryExpr(size_t tokenType) : ExprNode(tokenType) {}
    std::shared_ptr<ASTNode> getExpr();

    std::string toString() override;
};
