#pragma once

#include <string>
#include <memory>

#include "Symbol.h"

class Symbol; // forward declaration of Symbol to resolve circular dependency

class Scope {
public:
    virtual std::string getScopeName() = 0;

    /** Set the enclosing scope */
    virtual void setEnclosingScope(Scope* scope) = 0;

    /** Return the enclosing scope */
    virtual Scope* getEnclosingScope() = 0;

    /** Define a symbol in the current scope */
    virtual void define(Symbol* sym) = 0;

    /** Look up name in this scope or in enclosing scope if not here */
    virtual Symbol* resolve(const std::string &name) = 0;

    virtual std::string toString() = 0;
    virtual ~Scope() {};
};
