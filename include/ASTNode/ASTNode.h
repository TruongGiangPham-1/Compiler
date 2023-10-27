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
#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"

class Scope;
class Symbol;

class ASTNode {
public:
    antlr4::Token* token;                           // From which token did we create node?
    std::vector<std::shared_ptr<ASTNode>> children; // normalized list of children
    std::shared_ptr<Scope> scope;                   // containing scope

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

// instead of holding an ID node, we directly hold the symbol
// Children: [ IDNode, ExprNode ]
class AssignNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;

    AssignNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}

    std::shared_ptr<Symbol> getID();
    std::shared_ptr<ASTNode> getExprNode();
};

// Decl nodes are very similar to Assign nodes, but with more stuff
// Children: [ TypeNode, IDNode, ExprNode ]
class DeclNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;

    DeclNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ASTNode(tokenType), sym(sym) {}

    std::shared_ptr<ASTNode> getTypeNode();
    std::shared_ptr<Symbol> getIDNode();
    std::shared_ptr<ASTNode> getExprNode();
};

// ----
// EXPR
// ----

// all expressions must have a Type associated (this is populated in the Typecheck phase)
class ExprNode : public ASTNode {
public:
    std::shared_ptr<Type> type;  // For type checking

    ExprNode(size_t tokenType) : ASTNode(tokenType), type(nullptr) {}
    ExprNode(antlr4::Token* token) : ASTNode(token), type(nullptr) {}
};

// No children, just `sym` attribute
class IDNode : public ExprNode  {
public:
    std::shared_ptr<Symbol> sym; // pointer to symbol definition

    IDNode(antlr4::Token* token, std::shared_ptr<Symbol> sym) : ExprNode(token), sym(sym) {}
    IDNode(size_t tokenType, std::shared_ptr<Symbol> sym) : ExprNode(tokenType), sym(sym) {}

    std::string getName();
};

// `val` is calculated in the first pass (Builder)
class IntNode : public ExprNode {
public:
    int val;

    IntNode(size_t tokenType, int val) : ExprNode(tokenType), val(val) {}
    int getVal();
};

// Children: [leftExpr, rightExpr]
class BinaryExpr : public ExprNode
{
public:
    BINOP op;

    BinaryExpr(size_t tokenType) : ExprNode(tokenType) {}
    std::shared_ptr<ASTNode> getLHS();
    std::shared_ptr<ASTNode> getRHS();
private:
};

// Children: [expr]
class UnaryExpr : public ExprNode
{
public:
    UNARYOP op;

    UnaryExpr(size_t tokenType) : ExprNode(tokenType) {}
    std::shared_ptr<ASTNode> getExpr();
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
