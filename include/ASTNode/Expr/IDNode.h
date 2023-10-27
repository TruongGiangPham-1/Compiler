#include "ExprNode.h"


// No children, just `sym` attribute
class IDNode : public ExprNode  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(size_t tokenType, int line, std::shared_ptr<Symbol> sym) : ExprNode(tokenType, line), sym(sym) {}

    std::string toString() override;
    std::string getName();
};