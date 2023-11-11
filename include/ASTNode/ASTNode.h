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
protected:
    int line;                                       // line number in source file
public:
    std::vector<std::shared_ptr<ASTNode>> children; // normalized list of children
    std::shared_ptr<Scope> scope;                   // containing scope
    std::shared_ptr<Type> evaluatedType;
    ASTNode(); // for making nil-rooted nodes
    // create node with line number
    ASTNode(int line);

    void addChild(std::any t);
    void addChild(std::shared_ptr<ASTNode> t);

    /** Compute string for child nodes */
    virtual std::string toString();
    /** Compute string for a whole tree (only for top level node) */
    std::string toStringTree();

    size_t loc();

    virtual ~ASTNode();
};
