#pragma once

#include "antlr4-runtime.h"
#include "mlir/IR/Value.h"
#include "GazpreaParser.h"

#include <vector>
#include <string>
#include <memory>

#include "Scope.h"
#include "Type.h"
#include "Symbol.h"
#include "BINOP.h"

class Scope;
class Symbol;

class ASTNode {
public:
    antlr4::Token* token;       // From which token did we create node?
    std::vector<std::shared_ptr<ASTNode>> children; // normalized list of children
    std::shared_ptr<Scope> scope;               // containing scope

    ASTNode(); // for making nil-rooted nodes
    ASTNode(antlr4::Token* token);
    /** Create node from token type; used mainly for imaginary tokens */
    ASTNode(size_t tokenType);

    /** External visitors execute the same action for all nodes
     *  with same node type while walking. */
    size_t getNodeType();

    void addChild(std::any t);
    void addChild(std::shared_ptr<ASTNode> t);
    bool isNil();

    /** Compute string for single node */
    std::string toString();
    /** Compute string for a whole tree not just a node */
    std::string toStringTree();

    size_t loc();

    virtual ~ASTNode();
};

// TODO in Cymbol, the type node was the same as IDNode, and both were called SymAST
// I wonder if it's worth to differentiate between them...
class TypeNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type

    TypeNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ASTNode(token), sym(sym) {}
    TypeNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}
};

class ExprNode;
class AssignNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;
    std::shared_ptr<ExprNode> expr;

    AssignNode(size_t tokenType,std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}
};

// Decl nodes are very similar to Assign nodes, but with more stuff
class DeclNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<TypeNode> type;

    DeclNode(size_t tokenType,std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}
};

// ----
// EXPR
// ----


class ExprNode : public ASTNode {
public:
    std::shared_ptr<Type> type;  // For type checking

    ExprNode(size_t tokenType) : ASTNode(tokenType), type(nullptr) {}
    ExprNode(antlr4::Token* token) : ASTNode(token), type(nullptr) {}
};


class IDNode : public ExprNode  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ExprNode(token), sym(sym) {}
    IDNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ExprNode(tokenType), sym(sym) {}

    std::string getValue() { return sym->getName(); }
};

class IntNode : public ExprNode {
public:
    int val;

    IntNode(size_t tokenType, int val) : ExprNode(tokenType), val(val) {}
    int getValue() { return val; }
};


class BinaryExpr : public ExprNode
{
public:
    BinaryExpr(size_t tokenType) : ExprNode(tokenType), left(nullptr), right(nullptr) {}
    std::shared_ptr<ExprNode> getLHS() { return left; };
    std::shared_ptr<ExprNode> getRHS() { return right; };

    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    BINOP op;
private:
};

// a Unary expression
// TODO add unaryop enum
class UnaryExpr : public ExprNode
{
public:
    UnaryExpr(size_t tokenType) : ExprNode(tokenType), expr(nullptr) {}
    std::shared_ptr<ExprNode> getExpr() { return expr; }

    std::shared_ptr<ExprNode> expr;
};

class RangeVecNode : public BinaryExpr {
public:
    RangeVecNode(size_t tokenType) : BinaryExpr(tokenType) {}
};

class BinaryArithNode : public BinaryExpr {
public:
    BinaryArithNode(size_t tokenType) : BinaryExpr(tokenType){}
};

class BinaryCmpNode : public BinaryExpr {
public:
    BinaryCmpNode(size_t tokenType) : BinaryExpr(tokenType){}
};
