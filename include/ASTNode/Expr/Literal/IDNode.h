#pragma once
#include "ASTNode/Expr/ExprNode.h"

// No children, just `sym` attribute
class IDNode : public ExprNode  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(int line, std::shared_ptr<Symbol> sym) : ExprNode(line), sym(sym) {};

    std::string toString() override;
    std::string getName();

    std::shared_ptr<Symbol> getVal();
};
