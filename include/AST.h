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
