#include "ASTNode/Expr/Literal/TupleNode.h"

std::string TupleNode::toString() {
    return "Tuple";
}

std::vector<std::shared_ptr<ExprNode>> TupleNode::getVal() {
    return val;
}
