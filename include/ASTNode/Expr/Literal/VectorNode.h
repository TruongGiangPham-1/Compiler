//
// Created by Joshua Ji on 2023-11-13.
//

#ifndef GAZPREABASE_VECTORNODE_H
#define GAZPREABASE_VECTORNODE_H

#include "ASTNode/Expr/ExprNode.h"

// children: elements of the vector
class VectorNode : public ExprNode {
public:
    VectorNode(int line);

    std::vector<std::shared_ptr<ExprNode>> getElements();
    std::shared_ptr<ExprNode> getElement(int i);
    int getSize();
};


#endif //GAZPREABASE_VECTORNODE_H
