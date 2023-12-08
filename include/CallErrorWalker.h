//
// Created by Joshua Ji on 2023-12-07.
//

// CallErrorWalker is another pass that runs after def and ref
// the main goal is to catch CallErrors, such as procedure values being used in wrong circumstances

#ifndef GAZPREABASE_CALLERRORWALKER_H
#define GAZPREABASE_CALLERRORWALKER_H

#include "ContextedWalker.h"
#include "SyntaxWalker.h"
#include "SymbolTable.h"

namespace gazprea {
    class CallErrorWalker : public ContextedWalker {
    private:
        std::shared_ptr<SymbolTable> symTab;

        // true if it is a procedure, false if it is a function
        bool checkCallNodeType(const std::shared_ptr<CallNode>& tree);
    public:
        CallErrorWalker(std::shared_ptr<SymbolTable> symTab);

        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        std::any visitCall(std::shared_ptr<CallNode> tree) override;

        // === STREAMS
        std::any visitStreamOut(std::shared_ptr<StreamOut> tree) override;
    };
}


#endif //GAZPREABASE_CALLERRORWALKER_H
