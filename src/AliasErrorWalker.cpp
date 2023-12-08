//
// Created by Joshua Ji on 2023-12-08.
//

#include "AliasErrorWalker.h"

namespace gazprea {
    AliasErrorWalker::AliasErrorWalker() : ASTWalker() {}

    // true if the call symbol is a procedure
    bool AliasErrorWalker::isProcedure(const std::shared_ptr<CallNode>& tree) {
        if (std::dynamic_pointer_cast<FunctionSymbol>(tree->MethodRef)) {
            return false;
        } else if (std::dynamic_pointer_cast<ProcedureSymbol>(tree->MethodRef)) {
            return true;
        } else {
            throw std::runtime_error("AliasErrorWalker encountered something weird. See: AliasErrorWalker::checkCallNodeType method");
        }
    }

    std::any AliasErrorWalker::visitCall(std::shared_ptr<CallNode> tree) {
        // we don't care abt function calls, as their inputs are strictly const
        if (!isProcedure(tree)) return 0;

        // as procedures cannot be used as arguments in other procedures (this is checked in the CallError pass)
        // this reduces a LOT of complexity w.r.t alias checking (we don't have to check for aliases in the return types of procedures)
        // so instead we check the mlirName of all the variables passed into the procedure
        return 0;
    }
}