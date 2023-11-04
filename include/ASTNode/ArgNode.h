//
// Created by truong on 03/11/23.
//

#ifndef GAZPREABASE_ARGNODE_H
#define GAZPREABASE_ARGNODE_H

#include "ASTNode.h"
#include "Symbol.h"
#include "Types/QUALIFIER.h"
class ArgNode: public ASTNode {
public:
    ArgNode(int line);

    std::shared_ptr<Symbol>idSym;

    std::shared_ptr<ASTNode> getArgType();  // return child[0]
    std::string toString() override;
};

class ProcedureArgNode: public ArgNode {
public:
    QUALIFIER qualifier;
    ProcedureArgNode(int line): ArgNode(line) {};

    std::string toString() override {
      return "procedureArg";
    };

};

class FunctionArgNode: public ArgNode {
    FunctionArgNode(int line): ArgNode(line) {};
    std::string toString() override {
        return "functionArg";
    };
};
#endif //GAZPREABASE_ARGNODE_H
