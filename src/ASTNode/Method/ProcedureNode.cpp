//
// Created by truong on 03/11/23.
//

#include "ASTNode/Method/ProcedureNode.h"


ProcedureNode::ProcedureNode(int line, std::shared_ptr<Symbol> procedureNameSym): ASTNode(line), nameSym(procedureNameSym)  {

}


std::string ProcedureNode::toString() {
    if (body) {
        return "Procedure " + nameSym->getName() + " " + body->toStringTree();
    } else {
        return "Procedure " + nameSym->getName() + " forward";
    }
}

std::shared_ptr<ASTNode> ProcedureNode::getRetTypeNode() {
    if (this->children.size() > 0) {
        return this->children[0];
    } else {
        return nullptr;
    }
}
