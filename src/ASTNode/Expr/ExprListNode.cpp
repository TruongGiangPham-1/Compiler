#include "ASTNode/Expr/ExprListNode.h"

std::string ExprListNode::toString() {
    std::string returnString;
    for (unsigned i = 0; i < children.size(); i++) {
        if (i != children.size() - 1) {
            returnString += children[i]->toString() + ", ";
        }
        else {
            returnString += children[i]->toString();
        }
    }
    return returnString;
}
