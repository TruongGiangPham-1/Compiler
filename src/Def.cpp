//
// Created by truong on 28/10/23.
// NOTE: ALL THE DEF PASS WILL LOOK FOR GLOBAL DECLARATION / FORWARD DECLARATION
//
#include "../include/Def.h"
//#define DEBUG
namespace gazprea {
Def::Def(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirID) : symtab(symTab), varID(mlirID) {

    std::shared_ptr<GlobalScope> globalScope = std::make_shared<GlobalScope>();
    symTab->globalScope = globalScope;
    // push builtin type to global scope
    /*
     * populates the global scope with the mapping
     *    {baseType str, baseType Obj}
     *
     */
    globalScope->defineType(std::make_shared<AdvanceType>("integer"));
    globalScope->defineType(std::make_shared<AdvanceType>("real"));
    globalScope->defineType(std::make_shared<AdvanceType>("boolean"));
    globalScope->defineType(std::make_shared<AdvanceType>("character"));
    globalScope->defineType(std::make_shared<AdvanceType>("tuple"));
    globalScope->defineType(std::make_shared<AdvanceType>("string"));
    globalScope->defineType(std::make_shared<AdvanceType>("identity"));
    globalScope->defineType(std::make_shared<AdvanceType>("null"));

    // push all the builtin function symbol,make their return type int for now, can change in typecheck


    // simulate typdef  resolveType will walk up the type chain
    //globalScope->defineType(std::make_shared<AdvanceType>("integer", "quack"));
    //globalScope->defineType(std::make_shared<AdvanceType>("quack", "burger"));
    //globalScope->defineType(std::make_shared<AdvanceType>("burger", "chicken"));
    currentScope = symtab->enterScope(globalScope);  // enter global scope

    // define builtin functions
    defineBuiltins();
}

void Def::defineBuiltins() {
    auto resolvedInt = symtab->globalScope->resolveType("integer");

    auto streamState = std::make_shared<FunctionSymbol>("stream_state", "Global", symtab->globalScope->resolveType("integer"), symtab->globalScope, -1, true);
    auto length = std::make_shared<FunctionSymbol>("length", "global", resolvedInt, symtab->globalScope, -1, true);
    auto rows = std::make_shared<FunctionSymbol>("rows", "global", resolvedInt, symtab->globalScope, -1 ,true);
    auto columns = std::make_shared<FunctionSymbol>("columns", "global", resolvedInt, symtab->globalScope, -1, true);
    auto reverse = std::make_shared<FunctionSymbol>("reverse", "global", resolvedInt, symtab->globalScope, -1, true);
    auto format = std::make_shared<FunctionSymbol>("format", "global", resolvedInt, symtab->globalScope, -1, true);


    streamState->functypeENUM = FUNC_SSTATE;
    length->functypeENUM = FUNC_LENGTH;
    rows->functypeENUM = FUNC_ROW;
    columns->functypeENUM = FUNC_COLUMN;
    reverse->functypeENUM = FUNC_REVERSE;
    format->functypeENUM = FUNC_FORMAT;
    symtab->globalScope->define(streamState);
    symtab->globalScope->define(length);
    symtab->globalScope->define(rows);
    symtab->globalScope->define(columns);
    symtab->globalScope->define(reverse);
    symtab->globalScope->define(format);
    // EXAMPLE BUILTIN FUNCTION (TODO: delete once we have a proper stdlib)
    // define a vustom function silly() that returns an integer (no args)
    // to define args, use sillyDef->orderedArgs.push_back();

//     auto sillyDef = std::make_shared<FunctionSymbol>("silly", "Global", symtab->globalScope->resolveType("integer"), symtab->globalScope, -1, true);
//     sillyDef->orderedArgs.push_back(std::make_shared<Symbol>("arg1", resolvedInt, 0));
//     symtab->globalScope->define(sillyDef);
}


std::any Def::visitAssign(std::shared_ptr<AssignNode> tree) {
    return 0;
}

std::any Def::visitDecl(std::shared_ptr<DeclNode> tree) {
    return 0;
}

std::any Def::visitID(std::shared_ptr<IDNode> tree) {
    return 0;
}

std::any Def::visitTypedef(std::shared_ptr<TypeDefNode> tree) {
    // define type def mapping
    std::string typdefToString = tree->getName();
    symtab->defineTypeDef(tree->getType(), typdefToString, getNextId());

    auto typeN = std::dynamic_pointer_cast<TypeNode>(tree->getType());
    if (symtab->globalScope->typedefTypeNode.find(typeN->getTypeName()) != symtab->globalScope->typedefTypeNode.end()) {
        // this mapping already exists
        /*  eg
         * typedef integer[i] five;
         * typedef five poopoo;
         *  I will remove {five, integer[i]} mapping and will add {poopoo, integer[i]} mapping
         */
        auto topMostType = symtab->globalScope->typedefTypeNode[typeN->getTypeName()];  // this is gets integer[i] node
        symtab->globalScope->typedefTypeNode.erase(typeN->getTypeName());  // erase the old mapping
        symtab->globalScope->typedefTypeNode.emplace(typdefToString, topMostType);  // add new mapping
    } else {
        symtab->globalScope->typedefTypeNode.emplace(typdefToString, tree->getType());  // add new mapping
    }
    return 0;
}

std::any Def::visitProcedure(std::shared_ptr<ProcedureNode> tree) {

    // define if it is forwrd declaration

    if (tree->body) {
        // loop invariant, if it has body then forward declaration appears after this line, so it doesnt matter
        return 0;
    }  else {
        // forward declaration method
        std::shared_ptr<Type>retType;
        if (tree->getRetTypeNode()) {  // has return
            retType = symtab->resolveTypeUser(tree->getRetTypeNode());
            if (retType == nullptr) throw TypeError(tree->loc(), "cannot verify types");
        }
        std::string scopeName= "procScope" + tree->nameSym->getName() +std::to_string(tree->loc());
        std::shared_ptr<ScopedSymbol> methodSym = std::make_shared<ProcedureSymbol>(tree->nameSym->getName(),
                                                          scopeName, retType, symtab->globalScope, tree->loc());

        methodSym->typeSym = retType;
#ifdef DEBUG
        if (retType) {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " ret type "  << retType->getName() <<"\n";
        } else {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " no ret type \n";
        }
#endif
        // Here I am just adding argument symbol to the list of 'forwardDeclArgs' so that my Ref pass will be able to take this list and compare it with
        // method definition's arg for type check etc
        for (auto arg: tree->orderedArgs) {
            // define arg symbols
            methodSym->forwardDeclArgs.push_back(arg);
        }
        currentScope->define(methodSym);  // define methd symbol in global
//        defineFunctionAndProcedureArgs(tree->loc(), tree->nameSym, tree->orderedArgs, retType , 0);
//        currentScope = symtab->exitScope(currentScope);  // exit the argument scope
        assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));  // make sure we back to global scope
    }
    return 0;
}


std::any Def::visitFunction(std::shared_ptr<FunctionNode> tree) {
    if (tree->body || tree->expr) {  // we skip all function definition in def pass
        return 0;
    } else {
        // forward declaration method
        // I will just define the symbol
        std::shared_ptr<Type>retType;
        if (tree->getRetTypeNode()) {  // has return
            retType = symtab->resolveTypeUser(tree->getRetTypeNode());
            if (retType == nullptr) throw TypeError(tree->loc(), "cannot verify types");
        }
        std::string scopeName= "funcScope" + tree->funcNameSym->getName() +std::to_string(tree->loc());
        std::shared_ptr<ScopedSymbol> methodSym = std::make_shared<FunctionSymbol>(tree->funcNameSym->getName(),
                                                                                    scopeName, retType, symtab->globalScope, tree->loc());
        methodSym->typeSym = retType;
#ifdef DEBUG
        if (retType) {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " ret type "  << retType->getName() <<"\n";
        } else {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " no ret type \n";
        }
#endif
        // Here I am just adding argument nodes to the list of 'forwardDeclArgs' so that my Ref pass will be able to take this list and compare it with
        // method definition's arg for type check etc
        for (auto arg: tree->orderedArgs) {
            // define arg symbols
            methodSym->forwardDeclArgs.push_back(arg);
        }
        currentScope->define(methodSym);  // define methd symbol in global
//        defineFunctionAndProcedureArgs(tree->loc(), tree->nameSym, tree->orderedArgs, retType , 0);
//        currentScope = symtab->exitScope(currentScope);  // exit the argument scope
        assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));  // make sure we back to global scope
    }
    return 0;
}
std::any Def::visitCall(std::shared_ptr<CallNode> tree) {
    // SKIP in def pass
    return 0;
}

int Def::getNextId() {
    (*varID) ++;
    return *varID;
}
}
