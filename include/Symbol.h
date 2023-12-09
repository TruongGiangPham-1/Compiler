#pragma once

#include "Types/TYPES.h"
#include "mlir/IR/Value.h"

#include <string>
#include <memory>

#include "Type.h"
#include "Scope.h"
#include "Types/QUALIFIER.h"
#include "ASTNode/ASTNode.h"

class Scope; // forward declaration of Scope to resolve circular dependency
class ASTNode;
class Symbol { // A generic programming language symbol
public:
    std::string name;  // All symbols at least have a name
    std::shared_ptr<Scope> scope;
    std::shared_ptr<ASTNode> typeNode;
    QUALIFIER qualifier;
    TYPE type;
    std::shared_ptr<Type> typeSym;  // cast to advancedType!
    //
    int index = -1;  // for method parameters
    int functionStackIndex = -1;  // -1 indicates its in global
    // 2 indexes that stan requested
    int declarationIndex = -1;  // what is the order of declaration? even counts global scope
    int numStackBehind = -1;    // how many stack ago was this symbol declared from a reference? 0 means this symbol was declared in this scope
    int scopeDepthItWasDeclared = -1;  // num depth of the scope tree that it was declared at

    std::unordered_map<std::string , int> tupleIndexMap;  // look up map if the tuple index is an ID

    mlir::Value mlirAddr;
    std::string mlirName;

    Symbol(std::string name);
    Symbol(std::string name, std::shared_ptr<Type> type);
    Symbol(std::string name, std::shared_ptr<Type> type, std::shared_ptr<Scope> scope);

    // when defining builtin type parameters, we need to initialize a different set of parameters
    Symbol(std::string name, std::shared_ptr<Type> type, int index);

    virtual std::string getName();

    virtual std::string toString();

    // purely for debug purposes
    virtual std::string source();

    virtual ~Symbol();
};

class VariableSymbol : public Symbol {
public:
    VariableSymbol(std::string name, std::shared_ptr<Type> t) : Symbol(name, t) {}
    VariableSymbol(std::string name, std::shared_ptr<Type> t, std::shared_ptr<Scope> scope) : Symbol(name, t, scope) {}
};

