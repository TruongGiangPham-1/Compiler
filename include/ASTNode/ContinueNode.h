//
// Created by Joshua Ji on 2023-11-06.
//

#ifndef GAZPREABASE_CONTINUENODE_H
#define GAZPREABASE_CONTINUENODE_H

#include "ASTNode.h"

class ContinueNode : public ASTNode {
public:
    ContinueNode(int line);

    std::string toString() override;
};


#endif //GAZPREABASE_CONTINUENODE_H
