//
// Created by Joshua Ji on 2023-12-07.
//

#include "CallErrorWalker.h"

#include <utility>

namespace gazprea {
    CallErrorWalker::CallErrorWalker() : ContextedWalker() {}

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
                throw CallError(tree->loc(), "Procedure statement in an invalid context: " + contextToString(contexts.back()));
            }

            if (inContext(WALKER_CONTEXT::FUNCTION)) {
                throw CallError(tree->loc(), "Procedure statement in a function (Functions do not allow impure stmts)");
            }
        }
        if (tree->MethodRef == nullptr) {
            throw DefinitionError(tree->loc(), "attempting to call a function without definition");
        }

        contexts.push_back(WALKER_CONTEXT::INPUT_ARGS);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    // === EXPR ===
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

    std::any CallErrorWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
        contexts.push_back(WALKER_CONTEXT::BINOP);
        walkChildren(tree);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitVector(std::shared_ptr<VectorNode> tree) {
        contexts.push_back(WALKER_CONTEXT::VECTOR_LITERAL);
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

    // === FUNC STUFF ===
    std::any CallErrorWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
        contexts.push_back(WALKER_CONTEXT::FUNCTION);
        for (const auto &arg : tree->orderedArgs) {
            walk(arg);
        }
        if (tree->body) walk(tree->body);
        else if (tree->expr) walk(tree->expr);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
        contexts.push_back(WALKER_CONTEXT::PROCEDURE);
        for (const auto& arg : tree->orderedArgs) {
            walk(arg);
        }
        if (tree->body) walk(tree->body);
        contexts.pop_back();
        return 0;
    }

    std::any CallErrorWalker::visitReturn(std::shared_ptr<ReturnNode> tree) {
        contexts.push_back(WALKER_CONTEXT::RETURN_STMT);
        if (tree->getReturnExpr()) walk(tree->getReturnExpr());
        contexts.pop_back();
        return 0;
    }

    // === OTHER BLOCKS ===
    std::any CallErrorWalker::visitConditional(std::shared_ptr<ConditionalNode> tree) {
        contexts.push_back(WALKER_CONTEXT::CONDITIONAL_EXPR);
        for (const auto &condition : tree->conditions) {
            walk(condition);
        }
        contexts.pop_back();

        for (auto body : tree->bodies) {
            walk(body);
        }
        return 0;
    }

    std::any CallErrorWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
        contexts.push_back(WALKER_CONTEXT::CONDITIONAL_EXPR);
        walk(tree->getCondition());
        contexts.pop_back();

        walk(tree->getBody());
        return 0;
    }

    std::any CallErrorWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
        contexts.push_back(WALKER_CONTEXT::CONDITIONAL_EXPR);
        walk(tree->getCondition());
        contexts.pop_back();

        walk(tree->getBody());
        return 0;
    }

    std::any CallErrorWalker::visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree) {
        contexts.push_back(WALKER_CONTEXT::ITERATOR_DOMAIN);
        for (const auto &domainExpr : tree->getDomainExprs()) {
            walk(domainExpr.second);
        }
        contexts.pop_back();

        walk(tree->getBody());
        return 0;
    }
}
