#include "TypeNode.h"
#include "ASTNode/Expr/ExprNode.h"

class StringTypeNode : public TypeNode {
public:
    // size is nullptr if it is inferred
    std::shared_ptr<ASTNode> size;
    std::shared_ptr<ASTNode> innerType;

    // NOTES ON THE CONSTRUCTOR
    // we pass in a `sym` because TypeNode needs it
    // but it really isn't used I don't think
    // TODO find a better solution

    // initialize without a size
    StringTypeNode(int line, std::shared_ptr<Symbol> sym);
    // initialize with a size
    StringTypeNode(int line, std::shared_ptr<Symbol> sym, std::shared_ptr<ASTNode> size);

    /*
     * returns a **NULLABLE** ExprNode pointer for the size of the StringType
     * If size is inferred, returns nullptr
    */
    std::shared_ptr<ExprNode> getSize() const;
    /*
     * Returns true if the size of the vector is inferred
     */
    bool isInferred() const;

    std::string toString() override;
};
