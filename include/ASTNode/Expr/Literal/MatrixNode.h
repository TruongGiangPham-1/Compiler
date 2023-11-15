//
// Created by Joshua Ji on 2023-11-15.
//

#ifndef GAZPREABASE_MATRIXNODE_H
#define GAZPREABASE_MATRIXNODE_H

#include "ASTNode/Expr/ExprNode.h"
#include "VectorNode.h"

// children are the vectors that make up the matrix
class MatrixNode : public ExprNode {
public:
    MatrixNode(int line);

    std::vector<std::shared_ptr<VectorNode>> getElements();
    std::shared_ptr<VectorNode> getElement(int i);
    std::shared_ptr<ExprNode> getElement(int i, int j);

    // number of vectors
    int getRowSize();
    // size of each vector
    int getColSize();

    std::string toString() override;
};




#endif //GAZPREABASE_MATRIXNODE_H
