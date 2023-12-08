//
// Created by Joshua Ji on 2023-12-08.
//

#include "AliasErrorWalker.h"

#define DEBUG

namespace gazprea {
    AliasErrorWalker::AliasErrorWalker() : ASTWalker() {
        procedureArgIdx = -1;
    }

    std::any AliasErrorWalker::visitCall(std::shared_ptr<CallNode> tree) {
        procSymbol = std::dynamic_pointer_cast<ProcedureSymbol>(tree->MethodRef);
        if (!procSymbol) return 0; // we don't care about function calls

        // as procedures cannot be used as arguments in other procedures (this is checked in the CallError pass)
        // this reduces a LOT of complexity w.r.t alias checking (we don't have to check for aliases in the return types of procedures)
        // so instead we check the mlirName of all the variables passed into the procedure
        mlirNames.clear();
        procedureArgIdx = 0;
        for (const auto& arg: tree->children) {
            walk(arg);
            procedureArgIdx++;
        }
        procedureArgIdx = -1;
        return 0;
    }

    std::any AliasErrorWalker::visitID(std::shared_ptr<IDNode> tree) {
        if (procedureArgIdx < 0) return 0;
        if (tree->sym->qualifier == QUALIFIER::CONST) return 0; // is the variable is immutable, we do not care

        // what is the argument supposed to be?
        auto actualArgSym = procSymbol->orderedArgs.at(procedureArgIdx);
        assert(actualArgSym); // typechecker would have gotten invalid arg calls

        if (actualArgSym->qualifier == QUALIFIER::CONST) {
            // we only care about var args when checking for aliasing errors
            // since this argument is const, the value will be immutably passed
            return 0;
        }

        auto search = mlirNames.find(tree->sym->mlirName);
        if (search != mlirNames.end()) {
            throw AliasingError(tree->loc(), "Repeated alias with var " + tree->getName() + " in procedure call");
        } else {
            mlirNames.insert(tree->sym->mlirName);
        }
        return 0;
    }
}