#pragma once

#include "ASTNode/Expr/ExprNode.h"

// Children: [leftExpr, rightExpr]
// similar to ExprNode, this class should theoretically never be instantiated by itself
class BinaryExpr : public ExprNode
{
public:
    BINOP op;

    BinaryExpr(int line);
    std::shared_ptr<ExprNode> getLHS();
    std::shared_ptr<ExprNode> getRHS();

    // base toString method
    std::string toString() override;
};

// because these next two classes are so short, I'll keep them in this file
// they have no overrides and no new methods

class BinaryArithNode : public BinaryExpr {
public:
    BinaryArithNode(int line) : BinaryExpr(line){}
};

class BinaryCmpNode : public BinaryExpr {
public:
    BinaryCmpNode(int line) : BinaryExpr(line){}
};

class BinaryBoolNode : public BinaryExpr {
public:
    BinaryBoolNode(int line) : BinaryExpr(line){}
};

// given a[b], children are [a, b] , or a[b, b] children are [a, b, c]
class IndexNode : public BinaryExpr {
public:
    IndexNode(int line) : BinaryExpr(line){}
    std::shared_ptr<ExprNode> getIndexee();  // return a
    std::shared_ptr<ExprNode> getIndexor1();  // return b
    std::shared_ptr<ExprNode> getIndexor2();  // return c

};

class ConcatNode: public BinaryExpr {
public:
    ConcatNode(int line): BinaryExpr(line) {}
};
