//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_SCOPEDSYMBOL_H
#define GAZPREABASE_SCOPEDSYMBOL_H
#include "Symbol.h"
#include "Scope.h"
#include "FunctionCallTypes/FuncCallType.h"

class ScopedSymbol: public Symbol, public Scope{
protected:
    bool isBuiltin;
public:

    bool defined = false;
    FUNCTYPE functypeENUM = FUNCTYPE::FUNC_NORMAL;  // enum of buildIN
    std::shared_ptr<Scope>enclosingScope;
    std::vector<std::shared_ptr<Symbol>>orderedArgs;
    std::vector<std::pair<std::string, int>>declaredVars;  // all the variable declared in this mehtod

    std::vector<std::shared_ptr<ASTNode>>forwardDeclArgs;  // arguments of the forward declared

    ScopedSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope)
            : Symbol(symName, retType), enclosingScope(enclosingScope), isBuiltin(false) {};
    ScopedSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope, bool isBuiltin)
            : Symbol(symName, retType), enclosingScope(enclosingScope), isBuiltin(isBuiltin) {};
    // from Scope.h
    std::shared_ptr<Symbol> resolve(const std::string &name) override;
    void define(std::shared_ptr<Symbol> sym) override;
    std::shared_ptr<Scope> getEnclosingScope() override;
    void setEnclosingScope(std::shared_ptr<Scope> scope) override;
    std::shared_ptr<Type> resolveType(const std::string & name) override;
    std::string getScopeName() override {
        return "ScopeSymbol";
    };

    int incrementAndGetNumVarDeclared() override {
        this->numVarDeclared += 1;
        return this->numVarDeclared;
    };

    // from Symbol.h
    std::string getName() override;
    std::string toString() override;

    bool isBuiltIn() const;

};
// TODO: are they the same?
class FunctionSymbol: public ScopedSymbol {
public:
    std::string scopeName;
    int line;
    FunctionSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope, int line):
            ScopedSymbol(symName, scopeName, retType, enclosingScope), scopeName(scopeName), line(line) {};
    FunctionSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope, int line, bool isBuiltin):
            ScopedSymbol(symName, scopeName, retType, enclosingScope, isBuiltin), scopeName(scopeName), line(line) {};

    std::string getScopeName() override {
        return scopeName;
    };

};

class ProcedureSymbol: public ScopedSymbol {
public:
    std::string scopeName;
    int line;  // line number it was created

    ProcedureSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope, int line):
    ScopedSymbol(symName, scopeName, retType, enclosingScope), scopeName(scopeName), line(line) {};
    ProcedureSymbol(std::string symName, std::string scopeName, std::shared_ptr<Type> retType, std::shared_ptr<Scope> enclosingScope, int line, bool isBuiltin):
            ScopedSymbol(symName, scopeName, retType, enclosingScope, isBuiltin), scopeName(scopeName), line(line) {};

    std::string getScopeName() override {
        return  scopeName;
    };
};

#endif //GAZPREABASE_SCOPEDSYMBOL_H
