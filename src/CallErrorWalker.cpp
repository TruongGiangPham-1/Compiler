//
// Created by Joshua Ji on 2023-12-07.
//

#include "CallErrorWalker.h"

#include <utility>

namespace gazprea {
    CallErrorWalker::CallErrorWalker(std::shared_ptr<SymbolTable> symTab) : ContextedWalker() {
        this->symTab = std::move(symTab);
    }

    // true if the call symbol is a procedure
    // false if function
    bool CallErrorWalker::checkCallNodeType(const std::shared_ptr<CallNode>& tree) {
        if (std::dynamic_pointer_cast<FunctionSymbol>(tree->MethodRef)) {
            return false;
        } else if (std::dynamic_pointer_cast<ProcedureSymbol>(tree->MethodRef)) {
            return true;
        } else {
            throw std::runtime_error("CallErrorWalker encountered something weird. See: CallErrorWalker::isProcedure method");
        }
    }

    std::any CallErrorWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
        contexts.push_back(WALKER_CONTEXT::ASSIGN_BODY);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
        contexts.push_back(WALKER_CONTEXT::DECL_BODY);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitCall(std::shared_ptr<CallNode> tree) {
        // is this a function or a procedure?
        bool isProcedure = checkCallNodeType(tree);
        if (!isProcedure && tree->procCall) {
            // a call keyword is used on a function!
            throw CallError(tree->loc(), "Call statement used to call a function");
        }

        if (isProcedure) {
            bool validProcedureContext = directlyInContext(WALKER_CONTEXT::DECL_BODY)
                    || directlyInContext(WALKER_CONTEXT::ASSIGN_BODY);
            if (!validProcedureContext && !tree->procCall) {
                throw CallError(tree->loc(), "Procedure statement in an invalid context. Got: " + contextToString(contexts.back()));
            }
        }

        walkChildren(tree);
        return 0;
    }

    // === BINOP ===
    std::any CallErrorWalker::visitArith(std::shared_ptr<BinaryArithNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitIndex(std::shared_ptr<IndexNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitConcat(std::shared_ptr<ConcatNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitStride(std::shared_ptr<StrideNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }


    // === STREAM ===
    std::any CallErrorWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
        contexts.push_back(WALKER_CONTEXT::STREAM_OUT);
        walk(tree->getExpr());
        contexts.pop_back();
        return 0;
    }
}
