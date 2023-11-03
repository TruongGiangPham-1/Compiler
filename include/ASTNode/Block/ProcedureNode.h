//
// Created by truong on 03/11/23.
//

#ifndef GAZPREABASE_PROCEDURENODE_H
#define GAZPREABASE_PROCEDURENODE_H



#include "BlockNode.h"
#include "Symbol.h"
#include "Type.h"
class ProcedureNode : public BlockNode {
public:
    std::vector<std::shared_ptr<ASTNode>>orderedArgs;    // array of arguments's ID node
    std::shared_ptr<Symbol> nameSym;
    ProcedureNode(int line, std::shared_ptr<Symbol>procedureNameSym);
    std::string toString() override;
    std::shared_ptr<ASTNode> getRetTypeNode();
};



#endif //GAZPREABASE_PROCEDURENODE_H
