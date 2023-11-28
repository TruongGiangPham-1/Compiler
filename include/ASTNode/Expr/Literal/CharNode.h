#pragma once
#include "ASTNode/Expr/ExprNode.h"

// `val` is calculated in the first pass (Builder)
class CharNode : public ExprNode {
public:
    char val;

    CharNode(int line, char val) : ExprNode(line), val(val) {};

    std::string toString() override;
    char getVal();

    // Given a char after the slash, return the escape char (if valid)
    static std::optional<char> parseEscape(char c);

    static std::pair<char, std::string> consumeChar(std::string s);
};
