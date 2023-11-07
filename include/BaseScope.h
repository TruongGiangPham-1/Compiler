#pragma once

#include <map>
#include <string>

#include "Scope.h"
#include "AdvanceType.h"

class BaseScope : public Scope {
public:
    std::shared_ptr<Scope> enclosingScope; // nullptr if global (outermost) scope
    std::map<std::string, std::shared_ptr<Symbol>> symbols;  // add non types
    std::map<std::string,  std::shared_ptr<Symbol>> userTypes;  // mostly <string, AdvancedType*> members

    BaseScope(std::shared_ptr<Scope> parent) : enclosingScope(parent) {}

    std::shared_ptr<Symbol> resolve(const std::string &name) override;
    virtual void define(std::shared_ptr<Symbol> sym) override;
    virtual std::shared_ptr<Scope> getEnclosingScope() override;
    virtual void setEnclosingScope(std::shared_ptr<Scope> scope) override;

    virtual std::string toString() override;

    // for defining user types  type  i guess only globalScope should have this method to define All builtIn types and Typedefs
    void defineType(std::shared_ptr<Symbol>sym);
    virtual std::shared_ptr<Type> resolveType(const std::string& typeName) override;
};

class LocalScope : public BaseScope {
private:
    std::string scopeName;

public:
    LocalScope(std::string& sname, std::shared_ptr<Scope> parent)
            : BaseScope(parent), scopeName(sname) {}

    std::string getScopeName() override {
        return scopeName;
    }
};

class GlobalScope: public BaseScope {
public:
    GlobalScope(): BaseScope(nullptr) {};
    std::string getScopeName() override {
        return "Global";
    }
};

