//
// Created by truong on 28/10/23.
// NOTE: ALL THE DEF PASS WILL LOOK FOR GLOBAL DECLARATION / FORWARD DECLARATION
//
#include "../include/Def.h"

namespace gazprea {
Def::Def(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirID) : symtab(symTab), varID(mlirID) {

    std::shared_ptr<GlobalScope> globalScope = std::make_shared<GlobalScope>();
    symTab->globalScope = globalScope;
    // push builtin type to global scope
    globalScope->defineType(std::make_shared<AdvanceType>("integer", "integer"));
    globalScope->defineType(std::make_shared<AdvanceType>("real", "real"));
    globalScope->defineType(std::make_shared<AdvanceType>("boolean", "boolean"));
    globalScope->defineType(std::make_shared<AdvanceType>("character", "character"));
    globalScope->defineType(std::make_shared<AdvanceType>("tuple", "tuple"));
    globalScope->defineType(std::make_shared<AdvanceType>("matrix", "matrix"));
    globalScope->defineType(std::make_shared<AdvanceType>("string", "string"));

    // simulate typdef  resolveType will walk up the type chain
    //globalScope->defineType(std::make_shared<AdvanceType>("integer", "quack"));
    //globalScope->defineType(std::make_shared<AdvanceType>("quack", "burger"));
    //globalScope->defineType(std::make_shared<AdvanceType>("burger", "chicken"));
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
    //// resolve type
    std::shared_ptr<Type> resType = symtab->resolveTypeUser(tree->getTypeNode());

    assert(resType);  // ensure its not nullptr  // should be builtin type

    walk(tree->getExprNode());

    auto resolveID = currentScope->resolve(tree->getIDName());
    if (resolveID != nullptr) {
        throw SymbolError(tree->loc(), "redeclaration of identifier" + tree->getIDName());
    }

    // define the ID in symtable
    std::string mlirName = "VAR_DEF" + std::to_string(getNextId());
    std::shared_ptr<VariableSymbol> idSym = std::make_shared<VariableSymbol>(tree->getIDName(), resType);  //TODO: change TYPE to resolve type

    idSym->mlirName = mlirName;
    idSym->scope = currentScope;

    currentScope->define(idSym);

    std::cout << "line " << tree->loc() << " defined symbol " << idSym->name << " as type " << resType->getName()
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

std::any Def::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
    // define if it is forwrd declaration
    if (tree->body) {
        // not a forward decl, so we just skip it in def pass
        return 0;
    }  else {
        // forward declaration method
        std::shared_ptr<Type>retType;
        if (tree->hasReturn) {
            retType = symtab->resolveTypeUser(tree->getRetTypeNode());
        }
        tree->nameSym->typeSym = retType;  // set return type
        // define procedure scope Symbol
        std::string fname = "ProcedureScope" + tree->nameSym->getName() + std::to_string(tree->loc());
        std::shared_ptr<ProcedureSymbol> procSym = std::make_shared<ProcedureSymbol>(tree->nameSym->getName(),
                                                                                     fname, retType, symtab->globalScope, tree->loc());

        currentScope->define(procSym);  // define methd symbol in global
        std::cout << "defined method " << procSym->getName() << " in scope " << currentScope->getScopeName() << "\n";
        currentScope = symtab->enterScope( procSym);

        // define args
        for (auto argAST: tree->orderedArgs) {
            // define this myself, dont need mlir name because arguments are
            auto argNode = std::dynamic_pointer_cast<ArgNode>(argAST);
            //TODO: this id symbol dont have types yet. waiting for visitType implementation
            assert(argNode);  // not null
            assert(argNode->type);
            auto res= symtab->resolveTypeUser(argNode->type);
            if (res == nullptr) throw TypeError(tree->loc(), "unknown type ");
            argNode->idSym->typeSym = res;
            std::cout << "in line " << tree->loc()
                      << " argument = " << argNode->idSym->getName() << " defined in " << currentScope->getScopeName()
                      << " as type " << argNode->idSym->typeSym->getName() <<"\n";

            currentScope->define(argNode->idSym);  // define arg in curren scope
            argNode->scope = currentScope;  // set scope to function scope
        }
        currentScope = symtab->exitScope(currentScope);
        assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));  // make sure we back to global scope
    }
    return 0;
}


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
    if (tree->body) {
        return 0;
    } else {
        // TOODO: forward functino decl
    }
    return 0;
}
std::any Def::visitFunctionCall(std::shared_ptr<FunctionCallNode> tree) {
    // SKIP in def pass
    return 0;
}




int Def::getNextId() {
    (*varID) ++;
    return *varID;
}
}
