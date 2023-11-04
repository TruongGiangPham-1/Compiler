//
// Created by truong on 03/11/23.
//

#ifndef GAZPREABASE_PROCEDURENODE_H
#define GAZPREABASE_PROCEDURENODE_H



#include "BlockNode.h"
#include "Symbol.h"
#include "Type.h"
#include "CompileTimeExceptions.h"
class ProcedureNode : public BlockNode {
public:
    std::vector<std::shared_ptr<ASTNode>>orderedArgs;    // array of arguments's ID node
    std::shared_ptr<Symbol> nameSym;
    int hasReturn = 0;

    ProcedureNode(int line, std::shared_ptr<Symbol>procedureNameSym);
    std::string toString() override;
    std::shared_ptr<ASTNode> getRetTypeNode();

};


class ProcedureForwardNode: public  ProcedureNode {
public:
    ProcedureForwardNode(int line, std::shared_ptr<Symbol>procedureNameSym): ProcedureNode(line, procedureNameSym){};

    std::string toString() override {
        return "ProcedureForward";
    };
};

class ProcedureBlockNode: public ProcedureNode {
public:
    ProcedureBlockNode(int line, std::shared_ptr<Symbol>funcNameSym): ProcedureNode(line, funcNameSym){};
    std::shared_ptr<ASTNode> getBlock();
    std::string toString() override {
        return "ProcedureBlockNode";
    };
};

#endif //GAZPREABASE_PROCEDURENODE_H
