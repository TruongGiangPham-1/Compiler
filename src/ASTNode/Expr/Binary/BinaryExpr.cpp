#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "Operands/BINOP.h"

BinaryExpr::BinaryExpr(int line) : ExprNode(line) {};

std::string BinaryExpr::toString() {
    std::string opStr;

    switch (op) {
        case BINOP::ADD:
            opStr = "+";
                    break;
        case BINOP::DIV:
            opStr = "/";
            break;
        case BINOP::MULT:
            opStr = "*";
            break;
        case BINOP::SUB:
            opStr = "-";
            break;
        case BINOP::EQUAL:
            opStr = "==";
            break;
        case BINOP::NEQUAL:
            opStr = "!=";
            break;
        case BINOP::GTHAN:
            opStr = ">";
            break;
        case BINOP::LTHAN:
            opStr = "<";
            break;
    }
    return "BINOP " + opStr;
}

std::shared_ptr<ASTNode> BinaryExpr::getLHS() {
    return children[0];
}

std::shared_ptr<ASTNode> BinaryExpr::getRHS() {
    return children[1];
}
