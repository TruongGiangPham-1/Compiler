#include "TypeNode.h"
#include "ASTNode/Expr/ExprNode.h"

class TupleTypeNode : public TypeNode {
public:
    // a list of <ID?, TypeNode> pairs
    // if there is no ID at the tuple element, we just have an empty string
    std::vector<std::pair<std::string, std::shared_ptr<ASTNode>>> innerTypes;

    TupleTypeNode(int line, std::shared_ptr<Symbol> sym);

    /*
     * Returns the number of elements in the tuple
    */
    int numElements();

    /*
     * Returns the typenodes of the elements in a vector
    */
    std::vector<std::shared_ptr<TypeNode>> getTypes();

    /*
     * Returns the typeNode of the element at index or ID
     */
    std::shared_ptr<TypeNode> findType(int index);
    std::shared_ptr<TypeNode> findType(const std::string& id);

    std::string toString() override;
};
