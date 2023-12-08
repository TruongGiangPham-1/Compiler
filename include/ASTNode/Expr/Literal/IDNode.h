#pragma once
#include "ASTNode/Expr/ExprNode.h"

// No children, just `sym` attribute
class IDNode : public ExprNode  {
public:
    int numStackBehind = -1;    // how many stack ago was this ID declared from a reference? 0 means this symbol was declared in this scope
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(int line, std::shared_ptr<Symbol> sym) : ExprNode(line), sym(sym) {};

    std::string toString() override;
    std::string getName();

    std::shared_ptr<Symbol> getVal();
};
