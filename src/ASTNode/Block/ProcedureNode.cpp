//
// Created by truong on 03/11/23.
//



#include "../include/ASTNode/Block/ProcedureNode.h"



ProcedureNode::ProcedureNode(int line, std::shared_ptr<Symbol> procedureNameSym): BlockNode(line), nameSym(procedureNameSym)  {

}


std::string ProcedureNode::toString() {
    return "Procedure";
}

std::shared_ptr<ASTNode> ProcedureNode::getRetTypeNode() {
    // TODO: if procedure has no type, then children[0] = nullptr
    if (hasReturn) {
        return this->children[0];
    } else {
        throw CallError(1, "this procedure has no return value");
        return nullptr;
    }
}

// ===ProcedureBlock
std::shared_ptr<ASTNode> ProcedureBlockNode::getBlock() {
    if (hasReturn) {
        assert(this->children.size()==2);
        return this->children[1];
    } else {
        // if don't have return, then blockNode is at children[0]
        assert(this->children.size() == 1);
        return this->children[0];
    }
}

