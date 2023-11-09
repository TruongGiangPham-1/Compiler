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
    // define type def mapping
    std::string typdefToString = tree->getName();
    symtab->defineTypeDef(tree->getType(), typdefToString, getNextId());
    return 0;
}

std::any Def::visitProcedure(std::shared_ptr<ProcedureNode> tree) {

    // define if it is forwrd declaration
    if (symtab->globalScope->resolve(tree->nameSym->getName())) {
        throw SyntaxError(tree->loc(), "redefinition");
    }
    if (tree->body) {
        // loop invariant, if it has body then forward declaration appears after this line, so it doesnt matter
        return 0;
    }  else {
        // forward declaration method
        // swap
        auto find = this->prototype->find(tree->nameSym->getName());
        if (find == this->prototype->end())  return 0;
        auto methodDefAST = this->prototype->find(tree->nameSym->getName())->second;
        auto procedureDefAST = std::dynamic_pointer_cast<ProcedureNode>(tree);
        assert(procedureDefAST);
        if (tree->loc() < procedureDefAST->loc()) {
            // forward declare appear before method deifnietion so we swap body
            auto tempArg = tree->orderedArgs;
            tree->orderedArgs = procedureDefAST->orderedArgs;  // swap argument
            tree->body = procedureDefAST->body;               // swap body
            procedureDefAST->body = nullptr;
            procedureDefAST->orderedArgs = tempArg;          // swap arg

        }
        // erase the funcion definciton from the prototype map
        this->prototype->erase(tree->nameSym->getName());

        return 0;
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


    /*
     * @args:
     *      loc: line number
     *      funcNameSym = symbol of function/procedure name
     *      orderedArgs = list of function/prcedure arguments
     *      isFunc? 1: 0    // 1 to define function, 0 to define procedure
     * 1.defines function name in global
     * 2. push method scope , enter it, and define arguments inside it
     *
     */
    void Def::defineFunctionAndProcedureArgs(int loc, std::shared_ptr<Symbol>funcNameSym, std::vector<std::shared_ptr<ASTNode>> orderedArgs,std::shared_ptr<Type> retType , int isFunc) {
        // TODO: resolve return type.
        // define function scope Symbol
        std::shared_ptr<ScopedSymbol> methodSym;
        if (isFunc) {
            std::string fname = "FuncScope" + funcNameSym->getName() +std::to_string(loc);
            methodSym  = std::make_shared<FunctionSymbol>(funcNameSym->getName(),
                                                          fname, retType, symtab->globalScope, loc);
        } else {
            std::string fname = "ProcScope" + funcNameSym->getName() +std::to_string(loc);
            methodSym = std::make_shared<ProcedureSymbol>(funcNameSym->getName(),
                                                          fname, retType, symtab->globalScope, loc);
        }
        methodSym->typeSym = retType;
        if (retType) {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " ret type "  << retType->getName() <<"\n";
        } else {
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName() << " no ret type \n";
        }

        currentScope->define(methodSym);  // define methd symbol in global
        currentScope = symtab->enterScope(methodSym);

        // define the argument symbols
        int index = 0;
        for (auto &argIDNode: orderedArgs) {
            // define this myself, dont need mlir name because arguments are
            auto argNode = std::dynamic_pointer_cast<ArgNode>(argIDNode);
            //TODO: this id symbol dont have types yet. waiting for visitType implementation
            assert(argNode);  // not null
            assert(argNode->type);  // assert it exist

            argNode->idSym->mlirName =  "VAR_DEF" + std::to_string(getNextId());  // create new mlirname

            auto resType = symtab->resolveTypeUser(argNode->type);
            argNode->idSym->typeSym =  retType;
            if (resType == nullptr) throw TypeError(loc, "cannot resolve type");
            std::cout << "in line " << loc
                      << " argument = " << argNode->idSym->getName() << " defined in " << currentScope->getScopeName() <<
                      " as Type " << argNode->idSym->typeSym->getName() <<" as mlirname=" << argNode->idSym->mlirName  <<"\n";

            // define mlirname
            argNode->idSym->scope = currentScope;
            argNode->idSym->index = index;
            index ++;
            currentScope->define(argNode->idSym);  // define arg in curren scope
            argNode->scope = currentScope;  // set scope to function scope
        }
        //currentScope = symtab->exitScope(currentScope);
    }


int Def::getNextId() {
    (*varID) ++;
    return *varID;
}
void Def::setPrototype(std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ASTNode>>> &prototype) {
        this->prototype = prototype;
    }
}
