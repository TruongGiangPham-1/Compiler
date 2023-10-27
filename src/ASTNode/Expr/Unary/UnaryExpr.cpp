#include "ASTNode/Expr/Unary/UnaryExpr.h"

UnaryExpr::UnaryExpr(int line) : ExprNode(line) {};

std::shared_ptr<ASTNode> UnaryExpr::getExpr() {
    return children[0];
}

std::string UnaryExpr::toString() {
    std::string opStr;
    switch (op) {
        case UNARYOP::POSITIVE:
            opStr = "+";
            break;
        case UNARYOP::NEGATIVE:
            opStr = "-";
            break;
    }
    return "UNARYOP " + opStr;
}
