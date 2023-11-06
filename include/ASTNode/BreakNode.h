//
// Created by Joshua Ji on 2023-11-06.
//

#ifndef GAZPREABASE_BREAKNODE_H
#define GAZPREABASE_BREAKNODE_H

#include "ASTNode.h"

class BreakNode : public ASTNode {
public:
    BreakNode(int line);

    std::string toString() override;
};


#endif //GAZPREABASE_BREAKNODE_H
