#include "ExprNode.h"


// No children, just `sym` attribute
class IDNode : public ExprNode  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ExprNode(token), sym(sym) {}
    IDNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ExprNode(tokenType), sym(sym) {}

    std::string toString() override;
    std::string getName();
};