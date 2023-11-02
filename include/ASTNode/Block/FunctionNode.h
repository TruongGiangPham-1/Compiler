//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_FUNCTIONNODE_H
#define GAZPREABASE_FUNCTIONNODE_H

#include "BlockNode.h"
#include "Symbol.h"
#include "Type.h"
class FunctionNode : public BlockNode {
public:
    std::vector<std::shared_ptr<ASTNode>>orderedArgs;
    std::shared_ptr<Symbol> funcNameSym;
    FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym);
    std::string toString() override;


};



#endif //GAZPREABASE_FUNCTIONNODE_H
