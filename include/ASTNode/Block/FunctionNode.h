//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_FUNCTIONNODE_H
#define GAZPREABASE_FUNCTIONNODE_H

#include "BlockNode.h"

class FunctionNode : public BlockNode {
public:
    std::vector<ASTNode>orderedArgs;
    FunctionNode(int line);
    std::string toString() override;


};



#endif //GAZPREABASE_FUNCTIONNODE_H
