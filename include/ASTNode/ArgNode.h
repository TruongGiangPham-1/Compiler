//
// Created by truong on 03/11/23.
//

#ifndef GAZPREABASE_ARGNODE_H
#define GAZPREABASE_ARGNODE_H

#include "ASTNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "Symbol.h"
#include "Types/QUALIFIER.h"
class ArgNode: public ASTNode {
public:
    ArgNode(int line);

    std::shared_ptr<Symbol> idSym;
    std::shared_ptr<ASTNode> type;


    std::string toString() override;
};

#endif //GAZPREABASE_ARGNODE_H
