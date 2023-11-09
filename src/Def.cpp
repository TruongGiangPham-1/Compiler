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


    // simulate typdef  resolveType will walk up the type chain
    //globalScope->defineType(std::make_shared<AdvanceType>("integer", "quack"));
    //globalScope->defineType(std::make_shared<AdvanceType>("quack", "burger"));
    //globalScope->defineType(std::make_shared<AdvanceType>("burger", "chicken"));
    currentScope = symtab->enterScope(globalScope);  // enter global scope
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
    //  typdef type id;
    // define type def mapping
   // symtab->globalScope->defineType(std::make_shared<AdvanceType>(tree->getType()->getTypeName(), tree->getName()));
    std::string typdefToString = tree->getName();
    symtab->defineTypeDef(tree->getType(), typdefToString, getNextId());
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
#ifdef DEBUG
        std::cout << "defined method " << procSym->getName() << " in scope " << currentScope->getScopeName() << "\n";
#endif
        currentScope = symtab->enterScope( procSym);

        // define args
        for (auto argAST: tree->orderedArgs) {
            // define this myself, dont need mlir name because arguments are
            auto argNode = std::dynamic_pointer_cast<ArgNode>(argAST);
            assert(argNode);  // not null
            assert(argNode->type);
            auto res= symtab->resolveTypeUser(argNode->type);
            if (res == nullptr) throw TypeError(tree->loc(), "unknown type ");
            argNode->idSym->typeSym = res;
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
                      << " argument = " << argNode->idSym->getName() << " defined in " << currentScope->getScopeName()
                      << " as type " << argNode->idSym->typeSym->getName() <<"\n";
#endif
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

std::any Def::visitFunction(std::shared_ptr<FunctionNode> tree) {
    if (tree->body || tree->expr) {  // we skip all function definition in def pass
        return 0;
    } else {
        // TOODO: forward functino decl
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
