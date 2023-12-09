//
// Created by Joshua Ji on 2023-12-08.
//

#include "ProcedureCallArgWalker.h"

#define DEBUG

namespace gazprea {
    AliasCount::AliasCount(bool mut) {
        if (mut) {
            mutReferenced = true;
            constReferences = 0;
        } else {
            constReferences = 1;
            mutReferenced = false;
        }
    }

    void AliasCount::incrementConstRef() {
        constReferences++;
    }

    ProcedureCallArgWalker::ProcedureCallArgWalker() : ASTWalker() {
        procedureArgIdx = -1;
    }

    std::any ProcedureCallArgWalker::visitCall(std::shared_ptr<CallNode> tree) {
        procSymbol = std::dynamic_pointer_cast<ProcedureSymbol>(tree->MethodRef);
        if (!procSymbol) return 0; // we don't care about function calls

        // as procedures cannot be used as arguments in other procedures (this is checked in the CallError pass)
        // this reduces a LOT of complexity w.r.t alias checking (we don't have to check for aliases in the return types of procedures)
        // so instead we check the mlirName of all the variables passed into the procedure
        mlirNames.clear();
        procedureArgIdx = 0;
        for (const auto& arg: tree->children) {
            // what is the argument defined as?
            auto actualArgSym = procSymbol->orderedArgs.at(procedureArgIdx);
            assert(actualArgSym);

            if (!validMutArg(arg) && actualArgSym->qualifier == QUALIFIER::VAR) {
                throw TypeError(arg->loc(), "Invalid argument passed into variable arg slot");
            }

            walk(arg);
            procedureArgIdx++;
        }
        procedureArgIdx = -1;
        return 0;
    }

    void ProcedureCallArgWalker::checkAliasing(std::shared_ptr<Symbol> sym, int loc, std::string varName) {
        if (procedureArgIdx < 0) return;

        // what is the argument supposed to be?
        auto actualArgSym = procSymbol->orderedArgs.at(procedureArgIdx);
        assert(actualArgSym); // typechecker would have gotten invalid arg calls

        if (sym->qualifier == QUALIFIER::CONST) {
            // the variable is declared immutable
            // if it is being passed into a variable argument slot, we should throw an error
            // which error? I'm doing an AssignError
            if (actualArgSym->qualifier == QUALIFIER::VAR) {
                throw TypeError(loc, "Const var is being passed into variable procedure arg");
            }
        }

        auto idMlirName = sym->mlirName;
        auto search = mlirNames.find(idMlirName);

        if (actualArgSym->qualifier == QUALIFIER::CONST) {
            // since this argument is const, the value will be immutably passed
            if (search != mlirNames.end()) {
                // check if it has been mutably referenced
                // if it has, this is an error (if there is a mutable reference, we can't use it again in a const reference
                if (mlirNames[idMlirName]->mutReferenced) {
                    throw AliasingError(loc, "Repeated alias with var " + varName + " in procedure call (var then const)");
                } else {
                    // if not, increment constReference by 1
                    mlirNames[idMlirName]->incrementConstRef();
                }
            } else {
                mlirNames[idMlirName] = std::make_shared<AliasCount>(false);
            }
        } else {
            // arg is variable (mutable)
            if (search != mlirNames.end()) {
                // if there is already another argument, no matter if it's const or var, we will throw an error
                throw AliasingError(loc, "Repeated alias with var " + varName + " in procedure call");
            } else {
                mlirNames[idMlirName] = std::make_shared<AliasCount>(true);
            }
        }

    }

    std::any ProcedureCallArgWalker::visitID(std::shared_ptr<IDNode> tree) {
        checkAliasing(tree->sym, tree->loc(), tree->getName());
        return 0;
    }

    bool ProcedureCallArgWalker::validMutArg(std::shared_ptr<ASTNode> tree) {
        if (std::dynamic_pointer_cast<IDNode>(tree)) {
            return true;
        } else if (std::dynamic_pointer_cast<TupleIndexNode>(tree)) {
            return true;
        } else if (std::dynamic_pointer_cast<IndexNode>(tree)) {
            return true;
        }
        return false;
    }
}