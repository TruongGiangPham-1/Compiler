//
// Created by truong on 03/11/23.
//



#include "../include/ASTNode/Block/ProcedureNode.h"



ProcedureNode::ProcedureNode(int line, std::shared_ptr<Symbol> procedureNameSym): ASTNode(line), nameSym(procedureNameSym)  {

}


std::string ProcedureNode::toString() {
    return "Procedure";
}

std::shared_ptr<ASTNode> ProcedureNode::getRetTypeNode() {
    
}
