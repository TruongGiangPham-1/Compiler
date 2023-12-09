#pragma once

#include <string>
#include <memory>

#include "Symbol.h"

class Symbol; // forward declaration of Symbol to resolve circular dependency

class Scope {
public:
    int numVarDeclared = 0;  // use to keep track of number of variable declared in this scope
    virtual std::string getScopeName() = 0;

    /** Set the enclosing scope */
    virtual void setEnclosingScope(std::shared_ptr<Scope> scope) = 0;

    /** Return the enclosing scope */
    virtual std::shared_ptr<Scope> getEnclosingScope() = 0;

    /** Define a symbol in the current scope */
    virtual void define(std::shared_ptr<Symbol> sym) = 0;

    /** Look up name in this scope or in enclosing scope if not here */
    virtual std::shared_ptr<Symbol> resolve(const std::string &name) = 0;
    virtual std::shared_ptr<Type> resolveType(const std::string& name) = 0;

    virtual std::string toString() = 0;
    virtual int incrementAndGetNumVarDeclared() = 0;
    virtual ~Scope() {};
};
