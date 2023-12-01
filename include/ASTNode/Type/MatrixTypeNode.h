#pragma once
#include "TypeNode.h"
#include "ASTNode/Expr/ExprNode.h"

class MatrixTypeNode : public TypeNode {
public:
    // size is nullptr if it is inferred
    std::shared_ptr<ASTNode> sizeLeft;
    std::shared_ptr<ASTNode> sizeRight;
    std::shared_ptr<ASTNode> innerType;

    // NOTES ON THE CONSTRUCTOR
    // we pass in a `sym` because TypeNode needs it
    // but it really isn't used I don't think
    // TODO find a better solution

    // initialize without a size
    MatrixTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> innerType);

    /*
     * returns a **NULLABLE** ExprNode pointer for the sizes of the MatrixType
     * If size is inferred, returns nullptr
    */
    std::shared_ptr<ExprNode> getLeftSize() const;
    std::shared_ptr<ExprNode> getRightSize() const;
    /*
     * Gets the type of the elements in the vector
     */
    std::shared_ptr<TypeNode> getInnerType() const;

    std::string toString() override;
};
