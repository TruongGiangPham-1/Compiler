//
// Created by truong on 02/11/23.
//
#include "../include/Ref.h"

namespace gazprea {
    Ref::Ref(std::shared_ptr<SymbolTable> symTab, std::shared_ptr<int>mlirIDptr) : symtab(symTab), varID(mlirIDptr) {
        // globalscope aleady populated
        currentScope = symtab->enterScope(symTab->globalScope);  // enter global scope
    }

    std::any Ref::visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) {
        std::shared_ptr<Symbol> funcSym = currentScope->resolve(tree->funcNameSym->getName());
        if (funcSym == nullptr) {
            // there is no declaratioin for this function
            std::cout << "line " << tree->loc() << " no forward declaration, define function\n";
            // need to declare and define this function
            defineFunctionAndProcedure(tree->loc(), tree->funcNameSym, tree->orderedArgs, 1);

        } else {
            // check if this symbol is a function symbol
            if (std::dynamic_pointer_cast<FunctionSymbol>(funcSym)) {
                // there was a forward declaration
                std::cout << "resolved function definition " << funcSym->getName() << " at line: " << tree->loc() << " at scope "
                    << currentScope->getScopeName()<< std::endl;
                //TODO: iterate through all the ordered arguments of this function definition, and check if all the types in arguments are the same
                // as the type of arguments in declaration. Raise error if types are mismatching
                //currentScope = symtab->en

            } else {  // this is not a functionSymbol
                throw SymbolError(tree->loc(), "function same name as another identifier in the global scope");
            }
        }
        return 0;
    }

    std::any Ref::visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree) {
        std::shared_ptr<Symbol> funcSym = currentScope->resolve(tree->funcNameSym->getName());
        if (funcSym == nullptr) {
            // there is no declaratioin for this function
            std::cout << "line " << tree->loc() << " no forward declaration, define function\n";
            // define method scope and push. define method symbol
            defineFunctionAndProcedure(tree->loc(), tree->funcNameSym, tree->orderedArgs, 1);

            // push a local scope for function block,  to walk childre
            std::string sname = "functionScope" + std::to_string(tree->loc());
            currentScope = symtab->enterScope(sname, currentScope);

            // TODO: use getExpr() later when function node have types
            walk(tree->children[0]);  // ref all the symbol inside function block;

            currentScope = symtab->exitScope(currentScope);  // pop local scope
            currentScope = symtab->exitScope(currentScope);  // pop method scope

        } else {
            // check if this symbol is a function symbol
            if (std::dynamic_pointer_cast<FunctionSymbol>(funcSym)) {
                // there was a forward declaration and we found it
                std::cout << "resolved function definition " << funcSym->getName() << " at line: " << tree->loc() << " at scope "
                          << currentScope->getScopeName()<< std::endl;
                //TODO: iterate through all the ordered arguments of this function definition, and check if all the types in arguments are the same
                // as the type of arguments in declaration. Raise error if types are mismatching
                //currentScope = symtab->en
                defineFunctionAndProcedure(tree->loc(), tree->funcNameSym, tree->orderedArgs, 1);
                // push a local scope for function block,  to walk childre
                std::string sname = "functionScope" + std::to_string(tree->loc());
                currentScope = symtab->enterScope(sname, currentScope);

                // TODO: use getExpr() later when function node have types
                walk(tree->children[0]);  // ref all the symbol inside function block;

                currentScope = symtab->exitScope(currentScope);  // pop local scope
                currentScope = symtab->exitScope(currentScope);  // pop method scope
            } else {  // this is not a functionSymbol
                throw SymbolError(tree->loc(), "function same name as another identifier in the global scope");
            }
        }
        return 0;
    }


    std::any Ref::visitID(std::shared_ptr<IDNode> tree) {
        std::shared_ptr<Symbol> referencedSymbol;
        if (tree->scope == nullptr) {
            // this identifier is declared in a function/procedure block, so it was not visited by the the Def pass
            // we can reoslve using scope of this function/procedure Block.  [currenScope = localScope, currentScope->enclosingScope = Function/procedure Scope
            assert(std::dynamic_pointer_cast<ScopedSymbol>(currentScope->getEnclosingScope()));
            referencedSymbol = currentScope->resolve(tree->sym->getName());
            tree->scope = currentScope;

        } else {
            // this identifier is outside of function/procedure block, so it was visited by Def pass, has scope
            referencedSymbol =  tree->scope->resolve(tree->sym->getName());
        }
        if (referencedSymbol == nullptr) {
            std::cout << "in line " << tree->loc()
                      << " ref null\n"; // variable not defined
        } else {
            std::cout << "in line " << tree->loc() << " id=" << tree->sym->getName()
                      << "  ref " << referencedSymbol->mlirName << " in scope " << tree->scope->getScopeName() << "\n";
        }
        tree->sym = referencedSymbol;
    }

    /*
     * @args:
     *      loc: line number
     *      funcNameSym = symbol of function/procedure name
     *      orderedArgs = list of function/prcedure arguments
     *      isFunc? 1: 0    // 1 to define function, 0 to define procedure
     * 1.defines function name in global
     * 2. push method scope and define argument inside it
     *
     */
    void Ref::defineFunctionAndProcedure(int loc, std::shared_ptr<Symbol>funcNameSym, std::vector<std::shared_ptr<ASTNode>> orderedArgs, int isFunc) {
        // TODO: resolve type. cant resolve type yet since ASTBuilder havent updated visitType
        std::shared_ptr<Type> retType = std::make_shared<BuiltInTypeSymbol>("integer");  // create a random type for now

        // define function scope Symbol
        std::string fname = "FuncScope" + funcNameSym->getName() +std::to_string(loc);
        std::shared_ptr<FunctionSymbol> funcSym = std::make_shared<FunctionSymbol>(funcNameSym->getName(),
                                                                                   fname, retType, symtab->globalScope);
        currentScope->define(funcSym);  // define function symbol in global


        currentScope = symtab->enterScope(funcSym);
        // define the argument symbols
        for (auto argIDNode: orderedArgs) {
            // define this myself, dont need mlir name because arguments are
            auto idNode = std::dynamic_pointer_cast<IDNode>(argIDNode);
            //TODO: this id symbol dont have types yet. waiting for visitType implementation
            assert(idNode);  // not null
            idNode->sym->mlirName =  "VAR_DEF" + std::to_string(getNextId());  // create new mlirname
            std::cout << "in line " << loc
                      << " argument = " << idNode->sym->getName() << " defined in " << currentScope->getScopeName() << "\n";

            // define mlirname
            currentScope->define(idNode->sym);  // define arg in curren scope
            idNode->scope = currentScope;  // set scope to function scope
        }
        //currentScope = symtab->exitScope(currentScope);
    }

    int Ref::getNextId() {
        (*varID) ++;
        return (*varID);
    }
}



