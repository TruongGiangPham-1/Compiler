#pragma once

#include "mlir/IR/Value.h"

#include <string>
#include <memory>

#include "Type.h"
#include "Scope.h"
#include "ASTNode.h"

class Scope; // forward declaration of Scope to resolve circular dependency

class Symbol { // A generic programming language symbol
public:
    std::string name;  // All symbols at least have a name
    Type* type;
    Scope* scope;
    mlir::Value mlirAddr;

    Symbol(std::string name);
    Symbol(std::string name, Type* type);
    Symbol(std::string name, Type* type, Scope* scope);

    virtual std::string getName();

    virtual std::string toString();

    // purely for debug purposes
    virtual std::string source();

    virtual ~Symbol();
};

class VariableSymbol : public Symbol {
public:
    VariableSymbol(std::string name, Type* t) : Symbol(name, t) {}
    VariableSymbol(std::string name, Type* t, Scope* scope) : Symbol(name, t, scope) {}
};
