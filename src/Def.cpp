//
// Created by truong on 28/10/23.
//
#include "../include/Def.h"

namespace gazprea {
Def::Def(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirID) : symtab(symTab), varID(mlirID) {

    std::shared_ptr<GlobalScope> globalScope = std::make_shared<GlobalScope>();
    symTab->globalScope = globalScope;
    // push builtin type to global scope
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("integer"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("vector"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("character"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("real"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("tuple"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("matrix"));
    globalScope->define(std::make_shared<BuiltInTypeSymbol>("boolean"));


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
    std::shared_ptr<VariableSymbol> idSym = std::make_shared<VariableSymbol>(tree->getIDName(), TYPE::INTEGER);  //TODO: change TYPE to resolve type
    idSym->mlirName = mlirName;
    idSym->scope = currentScope;

    currentScope->define(idSym);

    std::cout << "line " << tree->loc() << " defined symbol " << idSym->name << " as type " << type->getName()
              << " as mlirNmae: " << mlirName << "\n" ;

    tree->scope = currentScope;
    tree->sym = std::dynamic_pointer_cast<Symbol>(idSym);
    return 0;
}

std::any Def::visitID(std::shared_ptr<IDNode> tree) {
    // only resolves and add scope information in AST for ref pass
    std::shared_ptr<Symbol> referencedSymbol = currentScope->resolve(tree->sym->getName());
    //if (referencedSymbol == nullptr) {
    //    std::cout << "in line " << tree->loc()
    //              << " ref null\n"; // variable not defined
    //} else {
    //    std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
    //              << "  ref " << referencedSymbol->mlirName << " Type is " << referencedSymbol->type->getName()
    //              << std::endl;
    //}
    tree->sym = referencedSymbol;
    tree->scope = currentScope;
    return 0;
}

/*
std::any Def::visitFilter(std::shared_ptr<FilterNode> tree) {

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
}*/

/*
std::any Def::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
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
*/

std::any Def::visitConditional(std::shared_ptr<ConditionalNode> tree) {

    for (auto condition : tree->conditions) {
      walk(condition);
    }
        // enter scope
    std::string sname = "loopcond" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);

    for (auto body: tree->bodies) {
        walk(body);
    }
    currentScope = symtab->exitScope(currentScope);
    return 0;
}

std::any Def::visitLoop(std::shared_ptr<LoopNode> tree) {
    walk(tree->condition);
    // enter scope
    std::string sname = "loopcond" + std::to_string(tree->loc());
    currentScope = symtab->enterScope(sname, currentScope);

    walk(tree->body);
    
    currentScope = symtab->exitScope(currentScope);
    return 0;
}

std::any Def::visitFunction(std::shared_ptr<FunctionNode> tree) {
    return 0;
}

//std::any Def::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
//    std::cout << "def pass visit procedure\n";
//    return 0;
//}


/*
std::any Def::visitFunctionForward(std::shared_ptr<FunctionForwardNode> tree) {
    std::cout << "visiting def function forward\n";
    // TODO: resolve type. cant resolve type yet since ASTBuilder havent updated visitType
    std::shared_ptr<Type> retType = std::make_shared<BuiltInTypeSymbol>("integer");  // create a random type for now

    // define function scope Symbol
    std::string fname = "FuncScope" + tree->funcNameSym->getName() + std::to_string(tree->loc());
    std::shared_ptr<FunctionSymbol> funcSym = std::make_shared<FunctionSymbol>(tree->funcNameSym->getName(),
                                                                               fname, retType, symtab->globalScope, tree->loc());

    currentScope->define(funcSym);  // define function symbol in global
    std::cout << "in line " << tree->loc()
              << " functionNamer= " << tree->funcNameSym->getName() << " defined in " << currentScope->getScopeName() << "\n";
    currentScope = symtab->enterScope( funcSym);
    // define the argument symbols
    for (auto argIDNode: tree->orderedArgs) {
        // define this myself, dont need mlir name because arguments are
        auto idNode = std::dynamic_pointer_cast<IDNode>(argIDNode);
        //TODO: this id symbol dont have types yet. waiting for visitType implementation
        assert(idNode);  // not null
        std::cout << "in line " << tree->loc()
                  << " argument = " << idNode->sym->getName() << " defined in " << currentScope->getScopeName() << "\n";

        currentScope->define(idNode->sym);  // define arg in curren scope
        idNode->scope = currentScope;  // set scope to function scope
    }

    currentScope = symtab->exitScope(currentScope);
    return 0;
}

std::any Def::visitProcedureForward(std::shared_ptr<ProcedureForwardNode> tree) {
    // define forward declaration if any
    std::shared_ptr<Type>retType;
    if (tree->hasReturn) {
        retType = resolveType(tree->getRetTypeNode());
    }
    tree->nameSym->type = retType;  // set return type
    // define procedure scope Symbol
    std::string fname = "FuncScope" + tree->nameSym->getName() + std::to_string(tree->loc());
    std::shared_ptr<ProcedureSymbol> procSym = std::make_shared<ProcedureSymbol>(tree->nameSym->getName(),
                                                                               fname, retType, symtab->globalScope, tree->loc());

    currentScope = symtab->enterScope( procSym);

    // define args
    for (auto argIDNode: tree->orderedArgs) {
        // define this myself, dont need mlir name because arguments are
        auto argNode = std::dynamic_pointer_cast<ArgNode>(argIDNode);
        //TODO: this id symbol dont have types yet. waiting for visitType implementation
        assert(argNode);  // not null

        // TODO: weird bug where resolve is returning wrong Type here but correct Type in
        //auto res= resolveType(argNode->getArgType());
        //argNode->idSym->type = res;
        std::cout << "in line " << tree->loc()
                  << " argument = " << argNode->idSym->getName() << " defined in " << currentScope->getScopeName();
                   //" as type " << argNode->idSym->type->getName() <<"\n";

        currentScope->define(argNode->idSym);  // define arg in curren scope
        argNode->scope = currentScope;  // set scope to function scope
    }
    currentScope = symtab->exitScope(currentScope);
    return 0;
}
*/


std::shared_ptr<Type> Def::resolveType(std::shared_ptr<ASTNode> t) {
    // type note
    std::shared_ptr<TypeNode> typeN = std::dynamic_pointer_cast<TypeNode>(t);
    std::cout << "Resolve Type: " << typeN->getTypeName() << std::endl;
    if (typeN == nullptr) {
        throw TypeError(t->loc(), "cannot cast to TypeNode");
    }
    // currently naive and cheap resolve
    std::shared_ptr<Type> ty;
    std::shared_ptr<Symbol> res;
    switch (typeN->typeEnum) {
        case TYPE::INTEGER:
            res = symtab->globalScope->resolve("integer");
        case TYPE::REAL:
            res = (symtab->globalScope->resolve("real"));
        case TYPE::TUPLE:
            res = (symtab->globalScope->resolve("tuple"));
        case TYPE::VECTOR:
            res = (symtab->globalScope->resolve("vector"));
        case TYPE::BOOLEAN:
            res = (symtab->globalScope->resolve("boolean"));
        case TYPE::CHAR:
            res = (symtab->globalScope->resolve("character"));
        case TYPE::MATRIX:
            res = (symtab->globalScope->resolve("matrix"));
        case TYPE::STRING:
            res = (symtab->globalScope->resolve("string"));
    }
    assert(res);
    ty = std::dynamic_pointer_cast<Type>(res);
    assert(ty);
    return ty;
    // TODO: handling user type?
}


int Def::getNextId() {
    (*varID) ++;
    return *varID;
}
}
