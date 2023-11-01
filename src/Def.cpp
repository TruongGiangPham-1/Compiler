//
// Created by truong on 28/10/23.
//
#include "../include/Def.h"

namespace gazprea {
Def::Def(std::shared_ptr<SymbolTable> symTab) : symtab(symTab) {

    std::shared_ptr<GlobalScope> globalScope = std::make_shared<GlobalScope>();
    symTab->globalScope = globalScope;
    // push builtin type to global scope
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("int"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("vector"));

    currentScope = symtab->enterScope(globalScope);  // enter global scope
}


std::any Def::visitAssign(std::shared_ptr<AssignNode> tree) {
    std::shared_ptr<Symbol> sym = currentScope->resolve(tree->getIDName());
    tree->scope = currentScope;
    tree->sym->scope = currentScope;
    tree->sym->mlirName = sym->mlirName;
    walkChildren(tree);
    return 0;
}

std::any Def::visitDecl(std::shared_ptr<DeclNode> tree) {
    // resolve type
    std::shared_ptr<Type> type = resolveType(tree->getTypeNode());
    assert(type);  // ensure its not nullptr  // should be builtin type

    walk(tree->getExprNode());

    // define the ID in symtable
    std::string mlirName = "VAR_DEF" + std::to_string(getNextId());
    std::shared_ptr<VariableSymbol> idSym = std::make_shared<VariableSymbol>(tree->getIDName(), type);
    idSym->mlirName = mlirName;
    idSym->scope = currentScope;

    currentScope->define(idSym);

    std::cout << "line " << tree->loc() << " defined symbol " << idSym->name << " as type " << idSym->type->getName()
              << " as mlirNmae: " << mlirName << "\n";

    tree->scope = currentScope;
    tree->sym = std::dynamic_pointer_cast<Symbol>(idSym);
    return 0;
}

std::any Def::visitID(std::shared_ptr<IDNode> tree) {
    std::shared_ptr<Symbol> referencedSymbol = currentScope->resolve(tree->sym->getName());
    if (referencedSymbol == nullptr) {
        std::cout << "in line " << tree->loc()
                  << " ref null\n"; // variable not defined
    } else {
        std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
                  << "  ref " << referencedSymbol->mlirName << " Type is " << referencedSymbol->type->getName()
                  << std::endl;
    }
    tree->sym = referencedSymbol;
    tree->scope = currentScope;
    return 0;
}

std::any Def::visitFilter(std::shared_ptr<FilterNode> tree) {
    /*
     * 1. RESOLVE THE VARIABLES IN THE <DOMAIN> 1st
     * 2. push scope, define <DOMAINVAR> into the new scope,
     */


    walk(tree->getVecNode());  // walk to the <domain> to resolve any ID in <domain>

    // create gen/filter scope
    std::string sname = "genfilter" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);
    // define domainVar as in this scope
    std::shared_ptr<VariableSymbol> domainVarSym = std::make_shared<VariableSymbol>(tree->domainVar,
                                                                                    std::make_shared<BuiltInTypeSymbol>("int"));

    domainVarSym->scope = currentScope;
    domainVarSym->mlirName = "VAR_DEF" + std::to_string(getNextId());
    currentScope->define(domainVarSym);  // define domain var symbol in this scope
    std::cout << "in line " << tree->loc()
              << "domainVar=" << tree->domainVar << " defined as "
              << domainVarSym->mlirName << std::endl;

    tree->scope = currentScope;  // any resolve in this scope will find the domainVar
    tree->domainVarSym = domainVarSym;

    walk(tree->getExpr());
    currentScope = symtab->exitScope(currentScope);

    return 0;
}

std::any Def::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
    /*
     * 1. RESOLVE THE VARIABLES IN THE <DOMAIN> 1st
     * 2. push scope, define <DOMAINVAR> into the new scope,
     */
    walk(tree->getVecNode());  // walk to the <domain> to resolve any ID in <domain>

    // create gen/filter scope
    std::string sname = "genfilter" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);
    // define domainVar as in this scope
    std::shared_ptr<VariableSymbol> domainVarSym = std::make_shared<VariableSymbol>(tree->domainVar,
                std::make_shared<BuiltInTypeSymbol>("int"));

    domainVarSym->scope = currentScope;
    domainVarSym->mlirName = "VAR_DEF" + std::to_string(getNextId());
    currentScope->define(domainVarSym);  // define domain var symbol in this scope
    std::cout << "in line " << tree->loc()
              << "domainVar=" << tree->domainVar << " defined as "
              << domainVarSym->mlirName << std::endl;

    tree->scope = currentScope;  // any resolve in this scope will find the domainVar
    tree->domainVarSym = domainVarSym;

    walk(tree->getExpr());
    currentScope = symtab->exitScope(currentScope);


    return 0;
}

std::any Def::visitConditional(std::shared_ptr<ConditionalNode> tree) {
    walk(tree->condition);
    // enter scope
    std::string sname = "loopcond" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);

    for (auto &stmt: tree->getStatements()) {
        walk(stmt);
    }
    currentScope = symtab->exitScope(currentScope);
    return 0;
}

std::any Def::visitLoop(std::shared_ptr<LoopNode> tree) {
    walk(tree->condition);
    // enter scope
    std::string sname = "loopcond" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);
    for (auto &stmt: tree->getStatements()) {
        walk(stmt);
    }
    currentScope = symtab->exitScope(currentScope);
    return 0;
}

std::shared_ptr<Type> Def::resolveType(std::shared_ptr<ASTNode> t) {
    // type note
    std::shared_ptr<TypeNode> typeN = std::dynamic_pointer_cast<TypeNode>(t);
    std::cout << "Resolve Type: " << typeN->getTypeName() << std::endl;
    if (typeN == nullptr) {
        std::cerr << "cannot cast to type node at line " << t->loc() << "\n";
        return nullptr;
    }
    if (typeN->getTypeName() != "int" && typeN->getTypeName() != "vector") {
        std::cerr << "type must be int or vector, invalid type at line " << t->loc() << "\n";
        return nullptr;
    }

    std::shared_ptr<Type> typeSym = std::dynamic_pointer_cast<Type>(
            (symtab->globalScope->resolve(typeN->getTypeName())));
    return typeSym;
}

int Def::getNextId() {
    this->varID++;
    return varID;
}
}
