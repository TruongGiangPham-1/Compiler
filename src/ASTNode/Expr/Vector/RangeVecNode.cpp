#include "ASTNode/Expr/Vector/RangeVecNode.h"

std::string RangeVecNode::toString() {
    return "RangeVec";
}

std::shared_ptr<ASTNode> RangeVecNode::getStart() {
    return children[0];
}

std::shared_ptr<ASTNode> RangeVecNode::getEnd() {
    return children[1];
}
