#pragma once
#include "TypeNode.h"
#include "ASTNode/Expr/ExprNode.h"

class VectorTypeNode : public TypeNode {
public:
    // size is nullptr if it is inferred
    // we store it as ASTNode so it's easier to set in the ASTBuilder
    // the methods cast them to appropriate Expr and TypeNodes
    std::shared_ptr<ASTNode> size;
    std::shared_ptr<ASTNode> innerType;

    // NOTES ON THE CONSTRUCTOR
    // we pass in a `sym` because TypeNode needs it
    // but it really isn't used I don't think
    // TODO find a better solution

    // initialize without a size
    VectorTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> innerType);
    // initialize with a size
    VectorTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> size, std::shared_ptr<ASTNode> innerType);

    /*
     * returns a **NULLABLE** ExprNode pointer for the size of the VectorType
     * If size is inferred, returns nullptr
    */
    std::shared_ptr<ExprNode> getSize() const;
    /*
     * Gets the type of the elements in the vector
     */
    std::shared_ptr<TypeNode> getInnerType() const;
    /*
     * Returns true if the size of the vector is inferred
     */
    bool isInferred() const;

    std::string getTypeName() {
        return sym->getName();
    };

    std::string toString() override;
};
