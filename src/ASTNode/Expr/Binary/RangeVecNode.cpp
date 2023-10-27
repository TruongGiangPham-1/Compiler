#include "ASTNode/Expr/Binary/RangeVecNode.h"

std::string RangeVecNode::toString() {
    return "RangeVec: " + getLHS()->toString() + " .. " + getRHS()->toString();
}
