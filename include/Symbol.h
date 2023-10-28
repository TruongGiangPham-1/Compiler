#pragma once

#include "mlir/IR/Value.h"

#include <string>
#include <memory>

#include "Type.h"
#include "Scope.h"
#include "ASTNode/ASTNode.h"

class Scope; // forward declaration of Scope to resolve circular dependency

class Symbol { // A generic programming language symbol
public:
    std::string name;  // All symbols at least have a name
    std::shared_ptr<Type> type;
    std::shared_ptr<Scope> scope;
    mlir::Value mlirAddr;
    std::string mlirName;

    Symbol(std::string name);
    Symbol(std::string name, std::shared_ptr<Type> type);
    Symbol(std::string name, std::shared_ptr<Type> type, std::shared_ptr<Scope> scope);

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
