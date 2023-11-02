//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_SCOPEDSYMBOL_H
#define GAZPREABASE_SCOPEDSYMBOL_H
#include "Symbol.h"
#include "Scope.h"

class ScopedSymbol: public Symbol, public Scope{
public:
    std::shared_ptr<Scope>enclosingScope;
    std::vector<std::shared_ptr<Symbol>>orderedArgs;

    ScopedSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope)
            : Symbol(symName, retType), enclosingScope(enclosingScope) {};
    // from Scope.h
    std::shared_ptr<Symbol> resolve(const std::string &name) override;
    void define(std::shared_ptr<Symbol> sym) override;
    std::shared_ptr<Scope> getEnclosingScope() override;
    void setEnclosingScope(std::shared_ptr<Scope> scope) override;
    std::string getScopeName() override {
        return "ScopeSymbol";
    };

    // from Symbol.h
    std::string getName() override;
    std::string toString() override;
};

class FunctionSymbol: public ScopedSymbol {
public:
    std::string scopeName;
    FunctionSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope):
            ScopedSymbol(symName, scopeName, retType, enclosingScope), scopeName(scopeName) {};

    std::string getScopeName() override {
        return scopeName;
    };

};

class ProcedureSymbol: public ScopedSymbol {
public:
    std::string scopeName;
    std::string getScopeName() override {
        return  scopeName;
    };
};
#endif //GAZPREABASE_SCOPEDSYMBOL_H
