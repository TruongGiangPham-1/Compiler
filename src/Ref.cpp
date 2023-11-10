//
// Created by truong on 02/11/23.
//
#include "../include/Ref.h"
//#define DEBUG
namespace gazprea {
    Ref::Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int> mlirIDptr) : symtab(symTab), varID(mlirIDptr) {
        // globalscope aleady populated
        currentScope = symtab->enterScope(symTab->globalScope);  // enter global scope
    }

    void Ref::printTupleType(std::shared_ptr<Type> ty) {
        if (ty->baseTypeEnum == TYPE::TUPLE) {
#ifdef DEBUG
            std::cout << "printinting tupleType ====\n";
            // print type of each child
            for (auto c: ty->tupleChildType) {
                std::cout << c.second->getName() << "\n";
            }
            std::cout << "finish printinting tupleType ====\n";
#endif
        } else {
        }
    }


    std::any Ref::visitFunction(std::shared_ptr<FunctionNode> tree) {
        /*
         * whenever we see a functin prototype, we add them to this map.
         * so that in if the definition is lower in the file than the prototype, we swap!
         */
        if (tree->body == nullptr && tree->expr == nullptr) {  // this is a function prototype
            if (this->funcProtypeList.find(tree->funcNameSym->getName()) == this->funcProtypeList.end()) {
                // first time seeing this prototype in the file
                this->funcProtypeList.emplace(tree->funcNameSym->getName(), tree);
            } else {
                throw SymbolError(tree->loc(), ":redeclaration of prootype method");
            }
            return 0;  // forward declaration node, we skip
        }
        auto funcSym = currentScope->resolve(tree->funcNameSym->getName());  // try to resolve procedure name
        if (funcSym == nullptr) {  //  can't resolve means that there was no forward declaration
            // no forward declaration
            // define method scope and push. define method symbol
            auto reType = symtab->resolveTypeUser(tree->getRetTypeNode());
            if (reType == nullptr) throw TypeError(tree->loc(), "cannot resolve function return type");

            defineFunctionAndProcedureArgs(tree->loc(), tree->funcNameSym, tree->orderedArgs, reType, 1);

            // push a local scope for function block,  to walk childre
            std::string sname = "functionScope" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);

            if (tree->body) {
                walk(tree->body);  // ref all the symbol inside function block;
            } else if (tree->expr) {
                walk(tree->expr);
            } else {
                // we should never reach here
                throw SymbolError(tree->loc(), ":weird this is the most important thing");
            }

            currentScope = symtab->exitScope(currentScope);  // pop local scope
            currentScope = symtab->exitScope(currentScope);  // pop method scope
            assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));

        } else {
            if (std::dynamic_pointer_cast<FunctionSymbol>(funcSym)) {
                // there was a forward declaration
                auto funcSymCast = std::dynamic_pointer_cast<FunctionSymbol>(funcSym);
#ifdef DEBUG
                std::cout << "resolved function definition " << funcSym->getName() << " at line: " << tree->loc()
                          << " at scope "
                          << currentScope->getScopeName() << std::endl;
#endif DEBUG

                std::shared_ptr<Type> retType = nullptr;
                if (tree->getRetTypeNode()) {
                    retType = symtab->resolveTypeUser(tree->getRetTypeNode());
                    if (retType == nullptr) throw TypeError(tree->loc(), "cannot resolve functin return type");
                }

                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
                // IMPORTANT: update the line number of the method symbol to be one highest
                funcSymCast->line = tree->loc() < funcSymCast->line ? tree->loc(): funcSymCast->line;
                //
                currentScope = symtab->enterScope(funcSymCast);     // enter the procedure symbol scope

                defineForwardFunctionAndProcedureArgs(tree->loc(), funcSymCast, tree->orderedArgs, retType);

                // push local scope for body
                std::string sname = "funcScope" + std::to_string(tree->loc());
                currentScope = symtab->enterScope(sname, currentScope);

                if (tree->body) {
                    walk(tree->body);  // ref all the symbol inside function block;
                } else if (tree->expr) {
                    walk(tree->expr);
                }
                currentScope = symtab->exitScope(currentScope);  // pop local scope
                currentScope = symtab->exitScope(currentScope);  // pop method scope
                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));

                // swap here?  // swap if line number is greater than prototypes
                auto find = this->funcProtypeList.find(funcSym->getName());
                if (find == this->funcProtypeList.end()) return 0;  // there was forward declaration but the prototype appear after this defintino
                auto protoType = this->funcProtypeList.find(funcSym->getName())->second;

                if (protoType->loc() < tree->loc()) {
#ifdef DEBUG
                    std::cout << "swapping prototype and function definition\n";
#endif DEBUG
                    // swap the prototype
                    if (tree->body) {
                        auto tempArg = protoType->orderedArgs;
                        protoType->orderedArgs = tree->orderedArgs;
                        tree->orderedArgs = tempArg;
                        protoType->body = tree->body;
                        tree->body = nullptr;
                    } else {
                        assert(tree->expr);
                        auto tempArg = protoType->orderedArgs;
                        protoType->orderedArgs = tree->orderedArgs;
                        tree->orderedArgs = tempArg;
                        protoType->expr = tree->expr;
                        tree->expr = nullptr;
                    }
                }
            } else {
                throw SymbolError(tree->loc(), ":function same name as another identifier in the global scope");
            }
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
            if (tree->loc() < (size_t) cast->line) {
                throw SymbolError(tree->loc(), "function " + cast->getName() + " not defined at this point");
            } else {
#ifdef DEBUG
                std::cout << "line: " << tree->loc() << " ref function call " << sym->getName() << "\n";
#endif
            }
            tree->scope = currentScope;
            // reference to the function Symbol that we are calling. can get all arguments using
            // tree->functionRef->orderedArgs
            tree->MethodRef = cast;
        } else if (std::dynamic_pointer_cast<ProcedureSymbol>(sym)) {
            // resolved to a procedure
            // any procedure call in an expression will trigger functionCall runle :(
            std::shared_ptr<ProcedureSymbol> cast = std::dynamic_pointer_cast<ProcedureSymbol>(sym);
            if (tree->loc() < (size_t) cast->line) {
                throw SymbolError(tree->loc(), ":procedure " + cast->getName() + " not defined at this point");
            } else {
#ifdef DEBUG
                std::cout << "line: " << tree->loc() << " ref procedure call " << sym->getName() << "\n";
#endif
            }
            tree->scope = currentScope;
            // reference to the function Symbol that we are calling. can get all arguments using
            // tree->functionRef->orderedArgs
            tree->MethodRef = cast;
        } else {
            // function call overshaddowed by a non function declaration above || function dont exist
            std::string errMSg = sym->getName() + " is not a function to be called. It is undefined or overshadowed"
                                                  "by another declaration above\n";
            throw SymbolError(tree->loc(), ":" + errMSg);
        }
        return 0;

    }

    std::any Ref::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
        /*
         * whenever we see a prototype prototype, we add them to this map.
         * so that in if the definition is lower in the file than the prototype, we swap!
         */
        if (tree->body == nullptr) {
            if (this->procProtypeList.find(tree->nameSym->getName()) == this->procProtypeList.end()) {
                // first time seeing this prototype in the file
                this->procProtypeList.emplace(tree->nameSym->getName(), tree);
            } else {
                throw SymbolError(tree->loc(), ":redeclaration of prootype method");
            }
            return 0;  // forward declaration node, we skip
        }


        auto procSym = currentScope->resolve(tree->nameSym->getName());  // try to resolve procedure name
        if (procSym == nullptr) {  //  can't resolve means that there was no forward declaration
            // no forward declaration
            // define method scope and push. define method symbol
            std::shared_ptr<Type> retType = nullptr;
            if (tree->getRetTypeNode()) {
                retType = symtab->resolveTypeUser(tree->getRetTypeNode());
                if (retType == nullptr) throw TypeError(tree->loc(), "cannot resolve procedure return type");
            }
            defineFunctionAndProcedureArgs(tree->loc(), tree->nameSym, tree->orderedArgs, retType, 0);

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
            // case there is a forrward declaration
            if (std::dynamic_pointer_cast<ProcedureSymbol>(procSym)) {
                // there was a forward declaration(method prototype)
                auto procSymCast = std::dynamic_pointer_cast<ProcedureSymbol>(procSym);
                std::cout << "resolved procedure definition " << procSym->getName() << " at line: " << tree->loc()
                          << " at scope "
                          << currentScope->getScopeName() << std::endl;

                std::shared_ptr<Type> retType = nullptr;
                if (tree->getRetTypeNode()) {
                    retType = symtab->resolveTypeUser(tree->getRetTypeNode());
                    if (retType == nullptr) throw TypeError(tree->loc(), "cannot resolve procedure return type");
                }
                // IMPORTANT: update the line number of the method symbol to be one highest
                procSymCast->line = tree->loc() < procSymCast->line ? tree->loc(): procSymCast->line;
                //
                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
                currentScope = symtab->enterScope(procSymCast);     // enter the procedure symbol scope

                defineForwardFunctionAndProcedureArgs(tree->loc(), procSymCast, tree->orderedArgs, retType);

                // push local scope for body
                std::string sname = "procScope" + std::to_string(tree->loc());
                currentScope = symtab->enterScope(sname, currentScope);

                if (tree->body) {
                    walk(tree->body);  // ref all the symbol inside function block;
                }
                currentScope = symtab->exitScope(currentScope);  // pop local scope
                currentScope = symtab->exitScope(currentScope);  // pop method scope
                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));

                // swap here?  // swap if line number is greater than prototypes =======================================
                auto find = this->procProtypeList.find(procSym->getName());
                if (find == this->procProtypeList.end()) return 0;  // this definition is higher in the file then forward declaration so we dont need to swap



                auto protoType = this->procProtypeList.find(procSym->getName())->second;
                if (protoType->loc() < tree->loc()) {
#ifdef DEBUG
                    std::cout << "swapping prototype and function definition\n";
#endif DEBUG
                    // swap
                    auto tempArg = protoType->orderedArgs;
                    protoType->orderedArgs = tree->orderedArgs;
                    tree->orderedArgs = tempArg;
                    protoType->body = tree->body;
                    tree->body = nullptr;
                }
            } else {
                throw SymbolError(tree->loc(), ":procedure same name as another identifier in the global scope");
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

        if (!tree->getTypeNode()) {
            if (!tree->getExprNode()) {
                // TODO: or if exprNode is identity or null
                throw SyntaxError(tree->loc(), "Inferred declaration is missing expression.");
            }
        }

        auto resolveID = currentScope->resolve(tree->getIDName());
        if (resolveID != nullptr) {
            throw SymbolError(tree->loc(), ":redeclaration of identifier " + tree->getIDName());
        }

        // define the ID in symtable
        std::string mlirName = "VAR_DEF" + std::to_string(getNextId());

        std::shared_ptr<VariableSymbol> idSym;
        if (tree->getTypeNode()) {
            std::shared_ptr<Type> resType = symtab->resolveTypeUser(tree->getTypeNode());
            if (resType == nullptr) throw TypeError(tree->loc(), "cannot resolve type");
            idSym = std::make_shared<VariableSymbol>(tree->getIDName(), resType);
#ifdef DEBUG
            std::cout << "line " << tree->loc() << " defined symbol " << idSym->getName() << " as type " << resType->getName() << " as mlirNmae: " << mlirName << "\n" ;
            printTupleType(resType);
#endif
        }
        //assert(type);  // ensure its not nullptr  // should be builtin type
        if (tree->getExprNode()) {
            walk(tree->getExprNode());
            if (!tree->getTypeNode()) {
                 idSym = std::make_shared<VariableSymbol>(tree->getIDName(), nullptr);
            }
        }

        idSym->mlirName = mlirName;
        idSym->scope = currentScope;
        idSym->qualifier = tree->qualifier;

        currentScope->define(idSym);

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
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
                      << " ref null\n"; // variable not defined
#endif
            throw SyntaxError(tree->loc(), "Undeclared variable " + tree->sym->getName());
        } else {
#ifdef DEBUG
            std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
                      << "  ref mlirName " << referencedSymbol->mlirName << " in scope " << tree->scope->getScopeName();
            if (std::dynamic_pointer_cast<ScopedSymbol>(referencedSymbol->scope)) {
                std::cout << " index of param=" << referencedSymbol->index;
            }
            std::cout << "\n";
#endif
        }
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
    void Ref::defineFunctionAndProcedureArgs(int loc, std::shared_ptr<Symbol> funcNameSym,
                                             std::vector<std::shared_ptr<ASTNode>> orderedArgs,
                                             std::shared_ptr<Type> retType, int isFunc) {
        // TODO: resolve return type.
        // define function scope Symbol
        std::shared_ptr<ScopedSymbol> methodSym;
        if (isFunc) {
            std::string fname = "FuncScope" + funcNameSym->getName() + std::to_string(loc);
            methodSym = std::make_shared<FunctionSymbol>(funcNameSym->getName(),
                                                         fname, retType, symtab->globalScope, loc);
        } else {
            std::string fname = "ProcScope" + funcNameSym->getName() + std::to_string(loc);
            methodSym = std::make_shared<ProcedureSymbol>(funcNameSym->getName(),
                                                          fname, retType, symtab->globalScope, loc);
        }
        methodSym->typeSym = retType;
        if (retType) {
#ifdef DEBUG
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName()
                      << " ret type " << retType->getName() << "\n";
#endif
        } else {
#ifdef DEBUG
            std::cout << "defined method " << methodSym->getName() << " in scope " << currentScope->getScopeName()
                      << " no ret type \n";
#endif
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

            argNode->idSym->mlirName = "VAR_DEF" + std::to_string(getNextId());  // create new mlirname

            auto resType = symtab->resolveTypeUser(argNode->type);
            argNode->idSym->typeSym =  resType;
            if (resType == nullptr) throw TypeError(loc, "cannot resolve type");
#ifdef DEBUG
            std::cout << "in line " << loc
                      << " argument = " << argNode->idSym->getName() << " defined in " << currentScope->getScopeName()
                      <<
                      " as Type " << argNode->idSym->typeSym->getName() << " as mlirname=" << argNode->idSym->mlirName
                      << "\n";
#endif

            // define mlirname
            argNode->idSym->scope = currentScope;
            argNode->idSym->index = index;
            index++;
            currentScope->define(argNode->idSym);  // define arg in curren scope
            argNode->scope = currentScope;  // set scope to function scope
        }
        //currentScope = symtab->exitScope(currentScope);
    }

    /*
     * we populate the argument of the method symbol and type check between the parameters of  method definition and
     * method prototype arguments
     */
    void Ref::defineForwardFunctionAndProcedureArgs(int loc, std::shared_ptr<ScopedSymbol> methodSym,
                                                    std::vector<std::shared_ptr<ASTNode>> orderedArgs,
                                                    std::shared_ptr<Type> retType) {

        assert(std::dynamic_pointer_cast<ScopedSymbol>(currentScope));
        methodSym->typeSym = retType;

        if (orderedArgs.size() != methodSym->forwardDeclArgs.size())
            throw DefinitionError(loc, "argument mismatch between forward decl and definition");
        int index = 0;  // argument index for stan
        //
        for (int i = 0; i < orderedArgs.size(); i++) {
            auto argNodeDef = std::dynamic_pointer_cast<ArgNode>(
                    orderedArgs[i]);  // argnode[i] for this method definiton
            auto argNodeFw = std::dynamic_pointer_cast<ArgNode>(
                    methodSym->forwardDeclArgs[i]);  // argnode[i] for forward decl

            assert(argNodeDef->type);
            assert(argNodeFw->type);

            auto argNodeDefType = symtab->resolveTypeUser(argNodeDef->type);
            auto argNodeFwType = symtab->resolveTypeUser(argNodeFw->type);

            // TYPECHECK ---------------------------------------
            if (argNodeDefType == nullptr || argNodeFwType == nullptr) {  // case: we could not resolve either
                throw TypeError(loc, "cannot resolve type");
            }
            parametersTypeCheck(argNodeDefType, argNodeFwType, loc);
            // TYPECHECK ---------------------------------------
            // add arguments to the methdd scope  and walk tree
            // define mlirname
            argNodeDef->idSym->index = index;
            argNodeDef->idSym->typeSym = argNodeDefType;
            argNodeDef->idSym->mlirName = "VAR_DEF" + std::to_string(getNextId());  // create new mlirname
            argNodeDef->scope = currentScope;  // set scope to function scope
            index++;
#ifdef DEBUG
            std::cout << "in line " << loc
                      << " argument = " << argNodeDef->idSym->getName() << " defined in "
                      << currentScope->getScopeName() <<
                      " as Type " << argNodeDef->idSym->typeSym->getName() << " as mlirname="
                      << argNodeDef->idSym->mlirName << "\n";
#endif
            currentScope->define(argNodeDef->idSym);  // define arg in curren scope
            assert(std::dynamic_pointer_cast<GlobalScope>(currentScope->getEnclosingScope()));
        }
    }

        /*
         * make sure the types are the same between 2 parameters
         * TODO: check for const
         */
    void Ref::parametersTypeCheck(std::shared_ptr<Type> type1, std::shared_ptr<Type> type2, int loc) {
        // TYPECHECK ---------------------------------------
        if (type1->baseTypeEnum != type2->baseTypeEnum) {  // TODO: tuple check
            throw TypeError(loc, "type mismatch between forward decl and definitino");
        } else if (type1->baseTypeEnum == TYPE::TUPLE && type2->baseTypeEnum == TYPE::TUPLE) {
            // iterate thru each tuple child and compare type
            for (int i = 0; i < type1->tupleChildType.size(); i++) {
                if (type1->tupleChildType[i]->baseTypeEnum != type2->tupleChildType[i]->baseTypeEnum) {
                    throw TypeError(loc, "type mismatch between tuples");
                }
            }
        }
        // TYPECHECK ---------------------------------------
    }

    std::any Ref::visitConditional(std::shared_ptr<ConditionalNode> tree) {
        return 0;
    }

    int Ref::getNextId() {
        (*varID)++;
        return (*varID);
    }
}


