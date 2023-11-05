#include "ASTNode/Block/ConditionalNode.h"

ConditionalNode::ConditionalNode(int line){}

std::string ConditionalNode::toString() {
    std::string ret = "Conditional";

    // number of blocks is never smaller than number of conditionals
    for (int i = 0; i < conditions.size(); i++) {
        ret += " (IF " + conditions[i]->toString() + " THEN " + bodies[i]->toString() + ")";
    }

    // if there is an else block
    if (bodies.size() > conditions.size()) {
        ret += " (ELSE " + bodies[bodies.size() - 1]->toString() + ")";
    }
    return ret;
}
