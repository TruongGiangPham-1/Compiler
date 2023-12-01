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

    std::any Ref::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) {
        // resolve id
        walk(tree->getIDNode());
        auto tupleNameNode = std::dynamic_pointer_cast<IDNode>(tree->getIDNode());


        auto tupleIDsym = currentScope->resolve(tupleNameNode->getName());
        if (tupleIDsym ) {
            // sometime it doesnt have typesymbol if declarated as var
            // declared tuple
            if (tupleIDsym->typeSym && tupleIDsym->typeSym->baseTypeEnum != TYPE::TUPLE) {
                throw SymbolError(tree->loc(), "cannot index non tuple");
            } else {
                tree->sym = tupleIDsym;


                if (std::dynamic_pointer_cast<IDNode>(tree->getIndexNode())) {
                    // index by ID
                    auto idCast =  std::dynamic_pointer_cast<IDNode>(tree->getIndexNode());
                    if (tupleIDsym->tupleIndexMap.find(idCast->getName()) == tupleIDsym->tupleIndexMap.end()) {  // mapint of tuple {ID: position}
                        // cant find it
                        throw SymbolError(tree->loc(), "this tupple id index not in tupple");
                    } else {
                        tree->index = tupleIDsym->tupleIndexMap[idCast->getName()];
                        //tree->sym->index = tree->index;
                    }
                } else {
                    // index by integer
                    assert(std::dynamic_pointer_cast<IntNode>(tree->getIndexNode()));
                    auto intCast =  std::dynamic_pointer_cast<IntNode>(tree->getIndexNode());
                    tree->index = intCast->getVal();
                    if (tupleIDsym->typeSym && ((tree->index > tupleIDsym->typeSym->tupleChildType.size()) || tree->index < 1)) {
                        // index out of bound(assume base i index
                        throw SymbolError(tree->loc(), "tuple index out of bound");
                    }
                    tree->index --;  // mae is base 0 index
                    tree->sym->index = tree->index;
                }
#ifdef DEBUG
                std::cout << " index tuple " << tupleIDsym->getName() << " at index=" << tree->index <<std::endl;
#endif DEBUG
            }
        } else {
            throw SymbolError(tree->loc(), "undeclared variable");
        }
        return 0;
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
            auto retType = symtab->resolveTypeUser(tree->getRetTypeNode());
            if (retType == nullptr) throw TypeError(tree->loc(), "cannot resolve function return type");

            potentiallySwapTypeDefNode(tree->getRetTypeNode(), tree);

            // ----------------------------------
            std::string scopeName= "funcScope" + tree->funcNameSym->getName() +std::to_string(tree->loc());
            std::shared_ptr<ScopedSymbol> methodSym = std::make_shared<FunctionSymbol>(tree->funcNameSym->getName(),
                                                                                       scopeName, retType, symtab->globalScope, tree->loc());
            methodSym->typeSym = retType;
            currentScope->define(methodSym);  // define methd symbol in global
            currentScope = symtab->enterScope(methodSym);     // enter the procedure symbol scope
            // ----------------------------------
            defineFunctionAndProcedureArgs(tree->loc(), methodSym, tree->orderedArgs, retType);

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

                defineFunctionAndProcedureArgs(tree->loc(), funcSymCast, tree->orderedArgs, retType);
                methodParamErrorCheck(funcSymCast->forwardDeclArgs, tree->orderedArgs, tree->loc());

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

                swapMethodBody(protoType->loc(), tree->loc(), std::dynamic_pointer_cast<ASTNode>(protoType), std::dynamic_pointer_cast<ASTNode>(tree));
            } else {
                throw SymbolError(tree->loc(), ":function same name as another identifier in the global scope");
            }
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

                potentiallySwapTypeDefNode(tree->getRetTypeNode(), tree);
            }
            // --------------------------------------
            std::string scopeName= "procScope" + tree->nameSym->getName() +std::to_string(tree->loc());
            std::shared_ptr<ScopedSymbol> methodSym = std::make_shared<ProcedureSymbol>(tree->nameSym->getName(),
                                                                                        scopeName, retType, symtab->globalScope, tree->loc());
            methodSym->typeSym = retType;
            currentScope->define(methodSym);  // define methd symbol in global
            currentScope = symtab->enterScope(methodSym);     // enter the procedure symbol scope
            // --------------------------------------
            defineFunctionAndProcedureArgs(tree->loc(), methodSym, tree->orderedArgs, retType);

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
                    potentiallySwapTypeDefNode(tree->getRetTypeNode(), tree);
                }
                // IMPORTANT: update the line number of the method symbol to be one highest
                procSymCast->line = tree->loc() < procSymCast->line ? tree->loc(): procSymCast->line;
                // --------------------------------------------------------------
                assert(std::dynamic_pointer_cast<GlobalScope>(currentScope));
                currentScope = symtab->enterScope(procSymCast);     // enter the procedure symbol scope
                // -------------------------------------------------------------
                defineFunctionAndProcedureArgs(tree->loc(), procSymCast, tree->orderedArgs, retType);  // define arguments
                methodParamErrorCheck(procSymCast->forwardDeclArgs, tree->orderedArgs, tree->loc());  // TYPECHECK
                // ----------------------------------------------------------------------
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
                swapMethodBody(protoType->loc(), tree->loc(), std::dynamic_pointer_cast<ASTNode>(protoType), std::dynamic_pointer_cast<ASTNode>(tree));
            } else {
                throw SymbolError(tree->loc(), ":procedure same name as another identifier in the global scope");
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
            // but if the function is a builtin function, skip this check
            if (tree->loc() < (size_t) cast->line && !cast->isBuiltIn()) {
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





    std::any Ref::visitDecl(std::shared_ptr<DeclNode> tree) {
        // this is declare statement defined in funciton/procedure. NOT in global scope
        // resolve type
        //std::shared_ptr<Type> type = resolveType(tree->getTypeNode());

        //if (tree->scope) {  // this Node already has a scope so its declared in  Def pass
        //    return 0;
        walkChildren(tree);  // walks typenode and expr node

        if (!tree->getTypeNode()) {
            if (!tree->getExprNode()) {
                // TODO: or if exprNode is identity or null
                throw SyntaxError(tree->loc(), "Inferred declaration is missing expression.");
            }
        }

        auto resolveID = currentScope->resolve(tree->getIDName());
        if (resolveID != nullptr) {
            if (resolveID->scope->getScopeName().find("iterator") == std::string::npos) {  // resolved ID is not in iterator scope then its error
                // this is resolved in the iterator domain var
                throw SymbolError(tree->loc(), ":redeclaration of identifier " + tree->getIDName());
            }
            // else, any Identifier same name as one defined in iterator loop is ok

        }

        // define the ID in symtable
        std::string mlirName = "VAR_DEF" + std::to_string(getNextId());

        std::shared_ptr<VariableSymbol> idSym;
        if (tree->getTypeNode()) {
            std::shared_ptr<Type> resType = symtab->resolveTypeUser(tree->getTypeNode());
            if (resType == nullptr) throw TypeError(tree->loc(), "cannot resolve type");

            potentiallySwapTypeDefNode(tree->getTypeNode(), tree);


            tree->getTypeNode()->evaluatedType = resType;
            idSym = std::make_shared<VariableSymbol>(tree->getIDName(), resType);
#ifdef DEBUG
            std::cout << "line " << tree->loc() << " defined symbol " << idSym->getName() << " as type " << resType->getName() << " as mlirNmae: " << mlirName << "\n" ;
            printTupleType(resType);
#endif
            // === For tuple indexing populate the map fr ==============
            if (resType->baseTypeEnum == TYPE::TUPLE) {
                // populate the index
                auto tupleChilds = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode())->innerTypes;  // vect<pair<ID, ASTNode>> for child
                for (int i = 0; i < tupleChilds.size(); i++) {
                    if (tupleChilds[i].first != "") {
                        // tuple child has an ID
                        std::string tupleChildID = tupleChilds[i].first;
                        idSym->tupleIndexMap.emplace(tupleChildID, i);
                    }
                }
            }
            // =====================================
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

        if (referencedSymbol == nullptr) {
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
                      << " ref null\n"; // variable not defined
#endif
            throw SymbolError(tree->loc(), "Undeclared variable ");
        } else {
#ifdef DEBUG
            std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
                      << "  ref mlirName " << referencedSymbol->mlirName << " in scope " << tree->scope->getScopeName();
            if (std::dynamic_pointer_cast<ScopedSymbol>(referencedSymbol->scope)) {
                std::cout << " index of param=" << referencedSymbol->index ;
            } else if (referencedSymbol->typeSym->baseTypeEnum == TYPE::TUPLE) {
                for (auto kv: referencedSymbol->tupleIndexMap) {
                    std::cout << " tupleIndex " << kv.first << "=" << kv.second << " ";
                }
            }
            std::cout << "\n";
#endif
        }

        tree->sym = referencedSymbol;

        return 0;
    }



    std::any Ref::visitConditional(std::shared_ptr<ConditionalNode> tree) {
        for (auto condition : tree->conditions) {
            walk(condition);
        }

        for (auto body: tree->bodies) {
            // enter scope
            std::string sname = "loopcond" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);
            walk(body);
            currentScope = symtab->exitScope(currentScope);
        }
        return 0;
    }

    std::any Ref::visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) {
        // enter scope
        std::string sname = "loopcond" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(sname, currentScope);
        walk(tree->getBody());
        currentScope = symtab->exitScope(currentScope);
        return 0;
    }

    std::any Ref::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
        walk(tree->getCondition());

        // enter scope
        std::string sname = "loopcond" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(sname, currentScope);
        walk(tree->getBody());
        currentScope = symtab->exitScope(currentScope);
        return 0;
    }

    std::any Ref::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
        walk(tree->getCondition());

        // enter scope
        std::string sname = "loopcond" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(sname, currentScope);
        walk(tree->getBody());
        currentScope = symtab->exitScope(currentScope);
        return 0;
    }

    std::any Ref::visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree) {
        // resolve the domain 1st
        for (auto &domainExpr: tree->getDomainExprs()) {
            auto domain = domainExpr.second;
            walk(domain);
        }

        auto scopeName = "iteratorLoop" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(scopeName, currentScope);

        // define domainVar
        auto intType = currentScope->resolveType("integer");  // domain var is just int right?
        for (auto &domainExpr: tree->getDomainExprs()) {
            auto domainVar = domainExpr.first;

            domainVar->mlirName = "VAR_DEF" + std::to_string(getNextId());
            domainVar->typeSym = intType;
            domainVar->scope = currentScope;
            currentScope->define(domainVar);
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
                      << "domainVar=" << domainV->getName() << " defined as "
                      << domainV->mlirName << std::endl;
#endif
        }
        // note, iterator variables in its own scope
        std::string sname = "loopcond" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(sname, currentScope);
        walk(tree->getBody());
        currentScope = symtab->exitScope(currentScope);
        currentScope = symtab->exitScope(currentScope);
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
    void Ref::defineFunctionAndProcedureArgs(int loc, std::shared_ptr<ScopedSymbol> methodSym,
                                             std::vector<std::shared_ptr<ASTNode>> orderedArgs,
                                             std::shared_ptr<Type> retType) {
        // TODO: resolve return type.
        // define function scope Symbol
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
        // define the argument symbols
        int index = 0;
        for (auto &argIDNode: orderedArgs) {
            // define this myself, dont need mlir name because arguments are
            auto argNode = std::dynamic_pointer_cast<ArgNode>(argIDNode);
            //TODO: this id symbol dont have types yet. waiting for visitType implementation
            assert(argNode);  // not null
            assert(argNode->type);  // assert it exist


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
            argNode->idSym->qualifier = argNode->qualifier;
            currentScope->define(argNode->idSym);  // define arg in curren scope
            argNode->scope = currentScope;  // set scope to function scope
            argNode->idSym->mlirName = "VAR_DEF" + std::to_string(getNextId());  // create new mlirname
            index++;
        }
        //currentScope = symtab->exitScope(currentScope);
    }

    void Ref::methodParamErrorCheck(std::vector<std::shared_ptr<ASTNode>> prototypeArg,
                                    std::vector<std::shared_ptr<ASTNode>> methodArg, int loc) {
        assert(std::dynamic_pointer_cast<ScopedSymbol>(currentScope));
        if (methodArg.size() != prototypeArg.size()) {
            throw DefinitionError(loc, "argument mismatch between forward decl and definition");
        }
        for (int i = 0; i < methodArg.size(); i++) {
            auto argNodeDef = std::dynamic_pointer_cast<ArgNode>(
                    methodArg[i]);  // argnode[i] for this method definiton
            auto argNodeFw = std::dynamic_pointer_cast<ArgNode>(
                    prototypeArg[i]);  // argnode[i] for forward decl
            assert(argNodeDef->type);
            assert(argNodeFw->type);

            auto argNodeDefType = symtab->resolveTypeUser(argNodeDef->type);
            auto argNodeFwType = symtab->resolveTypeUser(argNodeFw->type);
            parametersTypeCheck(argNodeDefType, argNodeFwType, loc);
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
                if (type1->tupleChildType[i].second->baseTypeEnum != type2->tupleChildType[i].second->baseTypeEnum) {
                    throw TypeError(loc, "type mismatch between tuples");
                }
            }
        }
        // TYPECHECK ---------------------------------------
    }


    void Ref::swapMethodBody(int prototypeLine, int methodDefinitionLine,
                             std::shared_ptr<ASTNode> prototypeNode, std::shared_ptr<ASTNode>tree) {
        if (std::dynamic_pointer_cast<ProcedureNode>(tree)) {
            if (prototypeLine < methodDefinitionLine) { // case: we want to swap procedure
#ifdef DEBUG
                std::cout << "swapping prototype and function definition\n";
#endif DEBUG
                auto protoType = std::dynamic_pointer_cast<ProcedureNode>(prototypeNode);  // cast it to procedure symbol
                auto defNode = std::dynamic_pointer_cast<ProcedureNode>(tree);
                assert(tree); assert(protoType);
                auto tempArg = protoType->orderedArgs;
                protoType->orderedArgs = defNode->orderedArgs;
                defNode->orderedArgs = tempArg;
                protoType->body = defNode->body;
                defNode->body = nullptr; }
        } else {
            assert(std::dynamic_pointer_cast<FunctionNode>(tree));  // case: we want to swap function
            // swap the prototype
            if (prototypeLine < methodDefinitionLine) {
#ifdef DEBUG
                std::cout << "swapping prototype and function definition\n";
#endif DEBUG
                auto protoType = std::dynamic_pointer_cast<FunctionNode>(prototypeNode);  // cast it to procedure symbol
                auto defNode = std::dynamic_pointer_cast<FunctionNode>(tree);
                assert(tree); assert(protoType);
                if (defNode->body) {
                    auto tempArg = protoType->orderedArgs;
                    protoType->orderedArgs = defNode->orderedArgs;
                    defNode->orderedArgs = tempArg;
                    protoType->body = defNode->body;
                    defNode->body = nullptr;
                } else {
                    assert(defNode->expr);
                    auto tempArg = protoType->orderedArgs;
                    protoType->orderedArgs = defNode->orderedArgs;
                    defNode->orderedArgs = tempArg;
                    protoType->expr = defNode->expr;
                    defNode->expr = nullptr;
                }
            }
        }
    }

    void Ref::potentiallySwapTypeDefNode(std::shared_ptr<ASTNode> typeNode, std::shared_ptr<ASTNode> tree) {
        /*
         * potentially swap to typedef node
         *
         */
        auto typeN = std::dynamic_pointer_cast<TypeNode>(typeNode);
        if (symtab->isTypeDefed(typeN->getTypeName())) {  // this type has typedef mapping
            tree->children[0] = symtab->globalScope->typedefTypeNode[typeN->getTypeName()];   // swap the real typenode here
        }
        return;
    }

    std::any Ref::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
        if (tree->domainVar1 == tree->domainVar2) throw  SymbolError(tree->loc(), "redefinition of domainVar");
        int isVec = 1;
        // walk the domain var to resolve them first
        if (tree->getVectDomain()) {
            // this is a vector generator
            walk(tree->getVectDomain());
        } else {
            // this is a matrix generator
            isVec = 0;
            auto domainPair = tree->getMatrixDomain();
            walk(domainPair.first);
            walk(domainPair.second);
        }

        auto scopeName = "generatorScope" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(scopeName, currentScope);

        // define domain var
        auto intType = currentScope->resolveType("integer");  // domain var is just int right?
        auto domainVar1Sym = std::make_shared<VariableSymbol>(tree->domainVar1, intType);
        auto domainVar2Sym = std::make_shared<VariableSymbol>(tree->domainVar2, intType);
        domainVar1Sym->scope = currentScope;
        domainVar1Sym->mlirName = "VAR_DEF" + std::to_string(getNextId());
        tree->scope = currentScope;
        if (isVec) {
            currentScope->define(domainVar1Sym);
            tree->domainVar1Sym = domainVar1Sym;
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
                      << "domainVar1=" << tree->domainVar1 << " defined as "
                      << domainVar1Sym->mlirName << std::endl;
#endif
        } else {
            domainVar2Sym->scope = currentScope;
            domainVar2Sym->mlirName = "VAR_DEF" + std::to_string(getNextId());
            currentScope->define(domainVar1Sym);
            currentScope->define(domainVar2Sym);
            tree->domainVar1Sym = domainVar1Sym;
            tree->domainVar2Sym = domainVar2Sym;
#ifdef DEBUG
            std::cout << "in line " << tree->loc()
              << "domainVar1=" << tree->domainVar1 << " defined as "
              << domainVar1Sym->mlirName <<  " domainVar2=" << tree->domainVar2 << " defined as "
              << domainVar2Sym->mlirName <<
              std::endl;
#endif
        }

        // walk expr
        walk(tree->getExpr());
        currentScope = symtab->exitScope(currentScope);

        return 0;
    }
    std::any Ref::visitFilter(std::shared_ptr<FilterNode> tree) {
        walk(tree->getDomain());

        auto scopeName = "filterScope" + std::to_string(tree->loc());
        currentScope = symtab->enterScope(scopeName, currentScope);
        // define domain var
        auto intType = currentScope->resolveType("integer");  // domain var is just int right?
        auto domainVarSym = std::make_shared<VariableSymbol>(tree->domainVar, intType);
        domainVarSym->scope = currentScope;
        domainVarSym->mlirName = "VAR_DEF" + std::to_string(getNextId());
        currentScope->define(domainVarSym);
        tree->domainVarSym = domainVarSym;
#ifdef DEBUG
        std::cout << "in line " << tree->loc()
                      << "domainVar=" << tree->domainVar << " defined as "
                      << domainVarSym->mlirName << std::endl;
#endif

        for (auto &expr: tree->getExprList()) {
            walk(expr);
        }

        currentScope = symtab->exitScope(currentScope);

        return 0;
    }




    int Ref::getNextId() {
        (*varID)++;
        return (*varID);
    }
}


