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

class AST {
public:
    antlr4::Token* token;       // From which token did we create node?
    std::vector<std::shared_ptr<AST>> children; // normalized list of children
    std::shared_ptr<Scope> scope;               // containing scope

    AST(); // for making nil-rooted nodes
    AST(antlr4::Token* token);
    /** Create node from token type; used mainly for imaginary tokens */
    AST(size_t tokenType);

    /** External visitors execute the same action for all nodes
     *  with same node type while walking. */
    size_t getNodeType();

    void addChild(std::any t);
    void addChild(std::shared_ptr<AST> t);
    bool isNil();

    /** Compute string for single node */
    std::string toString();
    /** Compute string for a whole tree not just a node */
    std::string toStringTree();

    size_t loc();

    virtual ~AST();
};

// TODO in Cymbol, the type node was the same as IDNode, and both were called SymAST
// I wonder if it's worth to differentiate between them...
class TypeNode : public AST {
public:
    std::shared_ptr<Symbol> sym; // symbol of the name of the type

    TypeNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : AST(token), sym(sym) {}
    TypeNode(size_t tokenType, std::shared_ptr<Symbol> sym) : AST(tokenType), sym(sym) {}
};

class ExprAST;
class AssignNode : public AST {
public:
    std::shared_ptr<Symbol> sym;
    std::shared_ptr<ExprAST> expr;

    AssignNode(size_t tokenType,std::shared_ptr<Symbol> sym) : AST(tokenType), sym(sym) {}
};

// Decl nodes are very similar to Assign nodes, but with more stuff
class DeclNode : public AST {
public:
    std::shared_ptr<Symbol> sym;
    std::shared_ptr<ExprAST> expr;
    std::shared_ptr<TypeNode> type;

    DeclNode(size_t tokenType,std::shared_ptr<Symbol> sym) : AST(tokenType), sym(sym) {}
};

// ----
// EXPR
// ----


class ExprAST : public AST {
public:
    std::shared_ptr<Type> type;  // For type checking

    ExprAST(size_t tokenType) : AST(tokenType), type(nullptr) {}
    ExprAST(antlr4::Token* token) : AST(token), type(nullptr) {}
};


class IDNode : public ExprAST  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ExprAST(token), sym(sym) {}
    IDNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ExprAST(tokenType), sym(sym) {}
};

class IntNode : public ExprAST {
public:
    int val;

    IntNode(size_t tokenType, int val) : ExprAST(tokenType), val(val) {}
};


class BinaryExpr : public ExprAST
{
public:
    BinaryExpr(size_t tokenType) : ExprAST(tokenType), left(nullptr), right(nullptr) {}
    std::shared_ptr<ExprAST> getLHS() { return left; };
    std::shared_ptr<ExprAST> getRHS() { return right; };

    std::shared_ptr<ExprAST> left;
    std::shared_ptr<ExprAST> right;
    BINOP op;
private:
};

// a Unary expression
// TODO add unaryop enum
class UnaryExpr : public ExprAST
{
public:
    UnaryExpr(size_t tokenType) : ExprAST(tokenType), expr(nullptr) {}
    std::shared_ptr<ExprAST> getExpr() { return expr; }

    std::shared_ptr<ExprAST> expr;
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
