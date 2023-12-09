//
// Created by Joshua Ji on 2023-12-07.
//

// CallErrorWalker is another pass that runs after def and ref
// the main goal is to catch CallErrors, such as procedure values being used in wrong circumstances

#ifndef GAZPREABASE_CALLERRORWALKER_H
#define GAZPREABASE_CALLERRORWALKER_H

#include "ContextedWalker.h"
#include "SymbolTable.h"
#include "SyntaxWalker.h"

namespace gazprea {
class CallErrorWalker : public ContextedWalker {
private:
    // true if it is a procedure, false if it is a function
    bool checkCallNodeType(const std::shared_ptr<CallNode>& tree);

public:
    CallErrorWalker();

    std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
    std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

    std::any visitCall(std::shared_ptr<CallNode> tree) override;

    // === EXPR
    std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
    std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
    std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
    std::any visitConcat(std::shared_ptr<ConcatNode> tree) override;
    std::any visitStride(std::shared_ptr<StrideNode> tree) override;
    std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;
    std::any visitVector(std::shared_ptr<VectorNode> tree) override;

    // === STREAMS
    std::any visitStreamOut(std::shared_ptr<StreamOut> tree) override;

    // === Function Stuff
    std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
    std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;
    std::any visitReturn(std::shared_ptr<ReturnNode> tree) override;

    // ohter blocks
    std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
    std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) override;
    std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) override;
    std::any visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree) override;
};
}

#endif // GAZPREABASE_CALLERRORWALKER_H
