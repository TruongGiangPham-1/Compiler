//
// Created by truong on 02/11/23.
//
#include "../include/Ref.h"

namespace gazprea {
    Ref::Ref(std::shared_ptr<SymbolTable> symTab) : symtab(symTab) {
        // globalscope aleady populated
        currentScope = symtab->enterScope(symTab->globalScope);  // enter global scope
    }

    std::any Ref::visitFunctionForward(std::shared_ptr<FunctionForwardNode> tree) {
        // skip
        return 0;
    }

    std::any Ref::visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) {
        std::shared_ptr<Symbol> funcSym = currentScope->resolve(tree->funcNameSym->getName());

        if (funcSym == nullptr) {
            // there is no declaratioin for this function
        } else {
            // check if this symbol is a function symbol
            if (std::dynamic_pointer_cast<FunctionSymbol>(funcSym)) {
                // there was a forward declaration
                std::cout << "resolved function name " << funcSym->getName() << " at line: " << tree->loc() << std::endl;
                // we need to push local scope to walk the children
                //currentScope = symtab->en

            } else {  // this is not a functionSymbol
                throw SymbolError(tree->loc(), "function same name as another identifier in the global scope");
            }
        }
        return 0;
    }

    std::any Ref::visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree) {
        return 0;
    }
}
