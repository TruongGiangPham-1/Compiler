//
// Created by truong on 02/11/23.
//
#include "../include/Ref.h"

namespace gazprea {
    Ref::Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr) : symtab(symTab), varID(mlirIDptr) {
        // globalscope aleady populated
        currentScope = symtab->enterScope(symTab->globalScope);  // enter global scope
    }


    std::any Ref::visitFunction(std::shared_ptr<FunctionNode> tree) {
        auto funcSym = currentScope->resolve(tree->funcNameSym->getName());  // try to resolve procedure name
        if (funcSym == nullptr) {  //  can't resolve means that there was no forward declaration
            // no forward declaration
            // define method scope and push. define method symbol
            auto reType = symtab->resolveTypeUser(tree->getRetTypeNode());
            if (reType == nullptr) throw TypeError(tree->loc(), "cannot resolve function return type");

            defineFunctionAndProcedureArgs(tree->loc(), tree->funcNameSym, tree->orderedArgs, reType,  1);

            // push a local scope for function block,  to walk childre
            std::string sname = "functionScope" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);

            if (tree->body) {
                walk(tree->body);  // ref all the symbol inside function block;
            } else if (tree->expr) {
                walk(tree->expr);
            } else {
                // we should never reach here
                throw SymbolError(tree->loc(), "weird this is the most important thing");
            }

            currentScope = symtab->exitScope(currentScope);  // pop local scope
            currentScope = symtab->exitScope(currentScope);  // pop method scope
            assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));

        }
        return 0;
    }
    std::any Ref::visitCall(std::shared_ptr<CallNode> tree) {
        walkChildren(tree);  // walk children to ref it
        std::shared_ptr<Symbol> sym;
        sym = currentScope->resolve(tree->CallName->getName());
        if (sym == nullptr) {
            throw SymbolError(tree->loc(), "Undefined call  " + tree->CallName->getName());
        }
        // you can get the ordered args using  cast->orderedArgs
        if (std::dynamic_pointer_cast<FunctionSymbol>(sym)) {  // resolved to a function
            // valid
            std::shared_ptr<FunctionSymbol> cast = std::dynamic_pointer_cast<FunctionSymbol>(sym);
            // check if it is called before declaration/definition
            if (tree->loc() < (size_t)cast->line) {
                throw SymbolError(tree->loc(), "function " + cast->getName() + " not defined at this point");
            } else {
                std::cout << "line: " << tree->loc() << " ref function call " << sym->getName() << "\n";
            }
            tree->scope = currentScope;
            // reference to the function Symbol that we are calling. can get all arguments using
            // tree->functionRef->orderedArgs
            tree->MethodRef = cast;
        } else if (std::dynamic_pointer_cast<ProcedureSymbol>(sym)) {
            // resolved to a procedure
            // any procedure call in an expression will trigger functionCall runle :(
            std::shared_ptr<ProcedureSymbol> cast = std::dynamic_pointer_cast<ProcedureSymbol>(sym);
            if (tree->loc() < (size_t)cast->line) {
                throw SymbolError(tree->loc(), "procedure " + cast->getName() + " not defined at this point");
            } else {
                std::cout << "line: " << tree->loc() << " ref procedure call " << sym->getName() << "\n";
            }
            tree->scope = currentScope;
            // reference to the function Symbol that we are calling. can get all arguments using
            // tree->functionRef->orderedArgs
            tree->MethodRef = cast;
        }

        else {
            // function call overshaddowed by a non function declaration above || function dont exist
            std::string errMSg = sym->getName() +  " is not a function to be called. It is undefined or overshadowed"
                                                   "by another declaration above\n";
            throw SymbolError(tree->loc(), errMSg);
        }
        return 0;

    }


    std::any Ref::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
        auto procSym = currentScope->resolve(tree->nameSym->getName());  // try to resolve procedure name
        if (procSym == nullptr) {  //  can't resolve means that there was no forward declaration
            // no forward declaration
            // define method scope and push. define method symbol
            std::shared_ptr<Type> retType = nullptr;
            if (tree->getRetTypeNode()) {
               retType = symtab->resolveTypeUser(tree->getRetTypeNode());
                if (retType == nullptr) throw TypeError(tree->loc(), "cannot resolve procedure return type");
            }
            defineFunctionAndProcedureArgs(tree->loc(), tree->nameSym, tree->orderedArgs, retType , 0);

            // push a local scope for function block,  to walk childre
            std::string sname = "procedureScope" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);

            if (tree->body) {
                walk(tree->body);  // ref all the symbol inside function block;
            }

            currentScope = symtab->exitScope(currentScope);  // pop local scope
            currentScope = symtab->exitScope(currentScope);  // pop method scope
            assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
        } else {
            if (std::dynamic_pointer_cast<ProcedureSymbol>(procSym)) {
                // there was a forward declaration
                // TODO:
            } else {
                throw SymbolError(tree->loc(), "procedure same name as another identifier in the global scope");
            }

        }
        return 0;
    }

    std::any Ref::visitDecl(std::shared_ptr<DeclNode> tree) {
        // this is declare statement defined in funciton/procedure. NOT in global scope
        // resolve type
        //std::shared_ptr<Type> type = resolveType(tree->getTypeNode());

        //if (tree->scope) {  // this Node already has a scope so its declared in  Def pass
        //    return 0;
        //}
        std::shared_ptr<Type> resType = symtab->resolveTypeUser(tree->getTypeNode());
        if (resType == nullptr) throw TypeError(tree->loc(), "cannot resolve type");

        //assert(type);  // ensure its not nullptr  // should be builtin type
        if (tree->getExprNode()) {
            walk(tree->getExprNode());
        }
        auto resolveID = currentScope->resolve(tree->getIDName());
        if (resolveID != nullptr) {
            throw SymbolError(tree->loc(), "redeclaration of identifier " + tree->getIDName());
        }

        // define the ID in symtable
        std::string mlirName = "VAR_DEF" + std::to_string(getNextId());
        std::shared_ptr<VariableSymbol> idSym = std::make_shared<VariableSymbol>(tree->getIDName(), resType);
        idSym->mlirName = mlirName;
        idSym->scope = currentScope;

        currentScope->define(idSym);

        std::cout << "line " << tree->loc() << " defined symbol " << idSym->getName() << " as type " << resType->getName() << " as mlirNmae: " << mlirName << "\n" ;

        tree->scope = currentScope;
        tree->sym = std::dynamic_pointer_cast<Symbol>(idSym);
        return 0;
    }

    std::any Ref::visitID(std::shared_ptr<IDNode> tree) {
        std::shared_ptr<Symbol> referencedSymbol;
        referencedSymbol = currentScope->resolve(tree->sym->getName());

        tree->scope = currentScope;
        tree->sym = referencedSymbol;

        if (referencedSymbol == nullptr) {
            std::cout << "in line " << tree->loc()
                      << " ref null\n"; // variable not defined
            throw SyntaxError(tree->loc(), "Undeclared variable " +  tree->sym->getName());
        } else {
            std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
                      << "  ref mlirName " << referencedSymbol->mlirName << " in scope " << tree->scope->getScopeName();
            if (std::dynamic_pointer_cast<ScopedSymbol>(referencedSymbol->scope)) {
                std::cout << " index of param=" << referencedSymbol->index ;
            }
            std::cout << "\n";
        }
        return 0;
    }

    /*
    std::any Ref::visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) {
        std::shared_ptr<Symbol> funcSym = currentScope->resolve(tree->funcNameSym->getName());
        if (funcSym == nullptr) {
            // there is no declaratioin for this function
            //std::cout << "line " << tree->loc() << " no forward declaration, define function\n";
            //// need to declare and define this function
            //defineFunctionAndProcedure(tree->loc(), tree->funcNameSym, tree->orderedArgs, 1);

            //// push a local scope for function block,  to walk childre
            //std::string sname = "functionScope" + std::to_string(tree->loc());
            //currentScope = symtab->enterScope(sname, currentScope);

            //// TODO: use getBody() later when function node have types
            //walk(tree->children[0]);  // ref all the symbol inside function block;

            //currentScope = symtab->exitScope(currentScope);  // pop local scope
            //currentScope = symtab->exitScope(currentScope);  // pop method scope
            //assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));

        } else {
            // check if this symbol is a function symbol
            if (std::dynamic_pointer_cast<FunctionSymbol>(funcSym)) {
                // there was a forward declaration
                //std::cout << "resolved function definition " << funcSym->getName() << " at line: " << tree->loc() << " at scope "
                //    << currentScope->getScopeName()<< std::endl;
                ////TODO: iterate through all the ordered arguments of this function definition, and check if all the types in arguments are the same
                //// as the type of arguments in declaration. Raise error if types are mismatching
                ////currentScope = symtab->en

                //// push a local scope for function block,  to walk childre
                //std::string sname = "functionScope" + std::to_string(tree->loc());
                //currentScope = symtab->enterScope(sname, currentScope);

                //// TODO: use getBody() later when function node have types
                //walk(tree->children[0]);  // ref all the symbol inside function block;

                //currentScope = symtab->exitScope(currentScope);  // pop local scope
                //currentScope = symtab->exitScope(currentScope);  // pop method scope
                //assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
            } else {  // this is not a functionSymbol
                throw SymbolError(tree->loc(), "function same name as another identifier in the global scope");
            }
        }
        return 0;
    }
    */



    // === Procedure

    /*
    std::any Ref::visitProcedureBlock(std::shared_ptr<ProcedureBlockNode> tree) {
        std::shared_ptr<Symbol> procSym = currentScope->resolve(tree->nameSym->getName());
        if (procSym == nullptr) {  // case: never been declared b4
            // define method scope and push. define method symbol
            defineFunctionAndProcedure(tree->loc(), tree->nameSym, tree->orderedArgs, 0);

            // push a local scope for function block,  to walk childre
            std::string sname = "procedureScope" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);

            walk(tree->getBlock());  // ref all the symbol inside function block;

            currentScope = symtab->exitScope(currentScope);  // pop local scope
            currentScope = symtab->exitScope(currentScope);  // pop method scope
            assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
        } else {
            // check if this symbol is a procedure symbol
            if (std::dynamic_pointer_cast<ProcedureSymbol>(procSym)) {  // CASE: function forward declared && valid function Symbol
                // there was a forward declaration and we found it
                std::cout << "resolved function definition " << procSym->getName() << " at line: " << tree->loc() << " at scope "
                          << currentScope->getScopeName()<< std::endl;
                defineFunctionAndProcedure(tree->loc(), tree->nameSym, tree->orderedArgs, 1);
                // push a local scope for function block,  to walk childre
                std::string sname = "functionScope" + std::to_string(tree->loc());
                currentScope = symtab->enterScope(sname, currentScope);

                //// TODO: use getExpr() later when function node have types
                walk(tree->getBlock());  // ref all the symbol inside function block;

                currentScope = symtab->exitScope(currentScope);  // pop local scope
                currentScope = symtab->exitScope(currentScope);  // pop method scope
                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));  // assert that we are at global scope for sanity
            } else {                                                     // case: conflicting identifier with another non function
                throw SymbolError(tree->loc(), "function same name as another identifier in the global scope");
            }
        }
        return 0;
    }
    */

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
    void Ref::defineFunctionAndProcedureArgs(int loc, std::shared_ptr<Symbol>funcNameSym, std::vector<std::shared_ptr<ASTNode>> orderedArgs,std::shared_ptr<Type> retType , int isFunc) {
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

    /*
    std::any Ref::visitFunction_call(std::shared_ptr<FunctionCallNode> tree) {
        std::shared_ptr<Symbol> sym;
        if (tree->functype == FUNCTYPE::FUNC_NORMAL) {
            sym = currentScope->resolve(tree->funcCallName->getName());
            assert(sym);
            std::shared_ptr<FunctionSymbol> cast = std::dynamic_pointer_cast<FunctionSymbol>(sym);
            if (cast) {
                // valid
                // check if it is called before declaration/definition
                if (tree->loc() < (size_t)cast->line) {

                   throw SymbolError(tree->loc(), "function " + cast->getName() + " not defined at this point");

                } else {

                    std::cout << "line: " << tree->loc() << " ref function call " << sym->getName() << "\n";

                }
            } else {
                // function call overshaddowed by a non function declaration above || function dont exist
                std::string errMSg = sym->getName() +  " is not a function to be called. It is undefined or overshadowed"
                                                       "by another declaration above\n";
                throw SymbolError(tree->loc(), errMSg);
            }
        }
        return 0;
    }
    */
    std::any Ref::visitConditional(std::shared_ptr <ConditionalNode> tree) {
        return 0;
    }

    int Ref::getNextId() {
        (*varID) ++;
        return (*varID);
    }
}



