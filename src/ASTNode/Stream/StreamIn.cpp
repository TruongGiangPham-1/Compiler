#include "ASTNode/Stream/StreamIn.h"
#include <memory>

StreamIn::StreamIn(int line) : ASTNode(line) {}

std::shared_ptr<ExprNode> StreamIn::getExpr() {
    return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::string StreamIn::toString() {
    return "StreamIn";
}
