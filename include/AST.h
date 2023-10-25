#pragma once

#include "antlr4-runtime.h"
#include "mlir/IR/Value.h"

#include <vector>
#include <string>
#include <memory>

#include "Scope.h"
#include "Type.h"
#include "Symbol.h"


class Scope;
class Symbol;

class AST {
public:
    antlr4::Token* token;       // From which token did we create node?
    std::vector<AST*> children; // normalized list of children
    Scope* scope;               // containing scope
    Symbol* symbol;             // containing symbol
    Type* type;
    mlir::Value mlirValue;

    AST(); // for making nil-rooted nodes
    AST(antlr4::Token* token);
    /** Create node from token type; used mainly for imaginary tokens */
    AST(size_t tokenType);

    /** External visitors execute the same action for all nodes
     *  with same node type while walking. */
    size_t getNodeType();

    void addChild(std::any t);
    void addChild(AST* t);
    bool isNil();

    /** Compute string for single node */
    std::string toString();
    /** Compute string for a whole tree not just a node */
    std::string toStringTree();

    size_t loc();

    virtual ~AST();
};

class IDNode : public AST {
public:
    Symbol* sym; // pointer to symbol definition

    IDNode(antlr4::Token* token, Symbol* sym) : AST(token), sym(sym) {}
    IDNode(size_t tokenType, Symbol* sym) : AST(tokenType), sym(sym) {}
};

class DeclNode : public AST {
public:
    Symbol* sym;
    AST* expr;

    DeclNode(antlr4::Token* token, Symbol* sym) : AST(token), sym(sym) {}
    DeclNode(size_t tokenType, Symbol* sym) : AST(tokenType), sym(sym) {}
};

// assign nodes have the same definition as DeclNodes
class AssignNode : public DeclNode {
public:
    AssignNode(antlr4::Token* token, Symbol* sym) : DeclNode(token, sym) {}
};

// ----
// EXPR
// ----


class ExprAST : public AST {
public:
    Type* type;  // For type checking

    ExprAST(antlr4::Token* token) : AST(token), type(nullptr) {}
    ExprAST(size_t tokenType) : AST(tokenType), type(nullptr) {}
};


class BinaryExpr
{
public:
    BinaryExpr(){}
    virtual ~BinaryExpr(){}
    virtual void getLHS() = 0;
    virtual void getRHS() = 0;
};

class UnaryExpr
{
public:
    UnaryExpr(){}
    virtual ~UnaryExpr(){}
    virtual void getExpr() = 0;
};

class RangeVecNode : public ExprAST, public BinaryExpr {
public:
    ExprAST* left;
    ExprAST* right;

    RangeVecNode(antlr4::Token* token) : ExprAST(token), left(nullptr), right(nullptr) {}
};

class BinaryArithNode : public ExprAST, public BinaryExpr {
public:
    ExprAST* left;
    ExprAST* right;

    BinaryArithNode(antlr4::Token* token) : ExprAST(token), left(nullptr), right(nullptr) {}
};
