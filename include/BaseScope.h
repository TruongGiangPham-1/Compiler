#pragma once

#include <map>
#include <string>

#include "Scope.h"

class BaseScope : public Scope {
public:
    Scope* enclosingScope; // nullptr if global (outermost) scope
    std::map<std::string, Symbol*> symbols;

    BaseScope(Scope *parent) : enclosingScope(parent) {}

    Symbol* resolve(const std::string &name) override;
    virtual void define(Symbol* sym) override;
    virtual Scope* getEnclosingScope() override;
    virtual void setEnclosingScope(Scope* scope) override;

    virtual std::string toString() override;
};

class LocalScope : public BaseScope {
private:
    std::string scopeName;

public:
    LocalScope(std::string& sname, Scope* parent)
            : BaseScope(parent), scopeName(sname) {}

    std::string getScopeName() override {
        return scopeName;
    }
};
