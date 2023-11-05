#include "ASTNode/Stream/StreamOut.h"
#include <memory>

StreamOut::StreamOut(int line) : ASTNode(line) {}

std::shared_ptr<ExprNode> StreamOut::getExpr() {
    return std::dynamic_pointer_cast<ExprNode>(children[0]);
}

std::string StreamOut::toString() {
    return "StreamOut";
}
