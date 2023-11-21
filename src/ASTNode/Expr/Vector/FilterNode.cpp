#include "ASTNode/Expr/Vector/FilterNode.h"

FilterNode::FilterNode(std::string domainVar, int line) : BaseVectorExpr(line), domainVar(domainVar) {}

std::string FilterNode::toString() {
    return "Filter " + domainVar;
}

std::shared_ptr<ASTNode> FilterNode::getDomain() {
    return children[0];
}

std::vector<std::shared_ptr<ASTNode>> FilterNode::getExprList() {
    std::vector<std::shared_ptr<ASTNode>> ret;  //  from josh's code in vectorNode.h
    ret.reserve(children.size() - 1);
    for (int i = 1; i < children.size(); i++) {  // ignore the first child cuz thats the domain expr
        ret.push_back(std::static_pointer_cast<ExprNode>(children[i]));
    }
    return ret;
}